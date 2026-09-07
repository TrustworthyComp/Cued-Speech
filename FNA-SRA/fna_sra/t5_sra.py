import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model, TaskType

from fna_sra.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from fna_sra.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from utils.evaluate_dict import DictionaryWER
from fna_sra.clip_loss import clip_loss
from fna_sra.asb import AbstractSLT
from transformers import get_cosine_schedule_with_warmup
from fna_sra.decoder_utils import PhonemeTokenizer
from fna_sra.ctc_align import CTCAlign
from fna_sra.transformer_align import TransformerAlign


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# torch.set_float32_matmul_precision('medium')


class SignerNorm(nn.Module):
    """
    Masked instance normalization for signer-invariant feature sequences.

    Normalises each sequence independently (per-sample, across the time axis)
    so that different signers' feature distributions are aligned before the
    temporal encoder.  Learnable affine parameters (weight / bias) let the
    model recover any useful scale after normalisation.
    """

    def __init__(self, feat_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(feat_dim))
        self.bias   = nn.Parameter(torch.zeros(feat_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) feature sequences (may contain padding)
            mask: (B, T) bool tensor — True for valid (non-padding) positions
        Returns:
            Normalised tensor of the same shape as x.
        """
        if mask is not None:
            valid = mask.float().unsqueeze(-1)                        # (B, T, 1)
            count = valid.sum(1, keepdim=True).clamp(min=1)          # (B, 1, 1)
            mean  = (x * valid).sum(1, keepdim=True) / count         # (B, 1, D)
            var   = ((x - mean).pow(2) * valid).sum(1, keepdim=True) / count
        else:
            mean = x.mean(1, keepdim=True)
            var  = x.var(1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


class GlobalCrossAttention(nn.Module):
    """
    Cross-attention where hand / lip features act as Query and global
    (full-frame) features act as Key and Value.

    Each hand / lip frame can selectively attend to all global frames to
    retrieve the spatial context (hand-to-face position) that is phonemically
    contrastive in Cued Speech but absent from tight region crops.

    A learnable gating scalar (initialised so sigmoid ≈ 0.12) keeps the
    global contribution small at the start of training, letting the model
    first learn the two-stream hand+lip baseline before opening the global
    context channel.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # sigmoid(-2) ≈ 0.12 — small initial gate so training is stable
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, T_q,  D) – hand or lip projected features
            key_value: (B, T_kv, D) – global projected features
            kv_mask:   (B, T_kv) bool – True = valid position
        Returns:
            (B, T_q, D) – enriched features (residual + gated cross-attn)
        """
        # nn.MultiheadAttention key_padding_mask convention: True → IGNORE
        kv_pad = (~kv_mask) if kv_mask is not None else None
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=kv_pad)
        return self.norm(query + torch.sigmoid(self.gate) * attn_out)


class TemporalAttentionPool(nn.Module):
    """
    Single-layer single-head temporal attention pooling.

    Replaces global average pooling with a learnable query vector that attends
    over the temporal dimension, producing a weighted sum of frame features.

    This verifies that the pooling strategy does not affect temporal modelling
    or recognition accuracy — i.e. single-head temporal attention and global
    average pooling are functionally equivalent for this aggregation step.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scale = hidden_dim ** 0.5
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) visual features
            mask: (B, T) bool tensor — True for valid (non-padding) positions
        Returns:
            (B, D) aggregated feature vector
        """
        B = x.shape[0]

        q = self.query.expand(B, -1, -1)                # (B, 1, D)
        k = self.key_proj(x)                             # (B, T, D)
        v = self.value_proj(x)                           # (B, T, D)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, T)

        # Apply mask: True=valid -> keep; False=pad -> -inf
        attn_mask = mask.float().unsqueeze(1)            # (B, 1, T)
        attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)           # (B, 1, T)

        return torch.bmm(attn_weights, v).squeeze(1)     # (B, D)


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model with multimodal capabilities.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
        lip_input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        eval_tokenizer: str = 'zh',   # 'zh' for Chinese, '13a' for English
        use_feature_norm: bool = False,   # Method 2: signer-invariant feature norm
        use_triplet_loss: bool = False,   # Method 3: cross-signer triplet/n-tuple loss
        triplet_margin: float = 1.0,      # Margin for triplet loss
        triplet_weight: float = 0.1,      # Weight of triplet loss in total loss
        use_supcon_loss: bool = False,    # Method 3b: SupCon replaces hard triplet
        supcon_temperature: float = 0.07, # Temperature for SupCon loss
        use_global_feat: bool = False,    # Enable global stream + cross-attention fusion
        global_cross_attn_heads: int = 8, # Number of attention heads in GlobalCrossAttention
        align_mode: str = 'clip',          # 'clip', 'ctc', or 'transformer'
        align_decoder_dim: int = 256,      # Hidden dim for CTC / Transformer decoder
        use_ta_pool: bool = False,         # Replace mean pooling with single-head temporal attention pooling
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.lip_input_size = lip_input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.eval_tokenizer = eval_tokenizer
        self.dict_wer_evaluator = DictionaryWER() if eval_tokenizer == 'phoneme_dict' else None
        self.use_feature_norm = use_feature_norm
        self.use_triplet_loss = use_triplet_loss
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.use_supcon_loss = use_supcon_loss
        self.supcon_temperature = supcon_temperature
        self.use_global_feat = use_global_feat
        self.global_cross_attn_heads = global_cross_attn_heads
        self.align_mode = align_mode
        self.align_decoder_dim = align_decoder_dim
        self.use_ta_pool = use_ta_pool

        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        self.set_container()
        
    # def load_pretrained_weights(self, checkpoint_path: str) -> None:
    #     """Load weights from a pretrained checkpoint."""
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
    #     # Get model's state dict
    #     model_state_dict = self.state_dict()
    #     checkpoint_state_dict = checkpoint['state_dict']
        
    #     # Filter out mismatched keys
    #     filtered_state_dict = {}
    #     for k, v in checkpoint_state_dict.items():
    #         if k in model_state_dict and v.size() == model_state_dict[k].size():
    #             filtered_state_dict[k] = v
        
    #     # Load the filtered state dict
    #     self.load_state_dict(filtered_state_dict)
    #     print(f'Checkpoint loaded from {checkpoint_path}. Loaded {len(filtered_state_dict)}/{len(checkpoint_state_dict)} parameters.')
    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint is loaded from {checkpoint_path}.')

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5 model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("LoRA adapter applied to T5 model.")

    def _freeze_model(self) -> None:
        """Freeze the T5 model parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5 model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str) -> None:
        """
        Prepare the textual and visual models.
        
        Args:
            t5_model: Name or path of the T5 model to use
        """
        
        # Load the textual model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16, 
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', self.lip_input_size, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        # Load the temporal encoder
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        # Global (full-frame) stream: projector + cross-attention modules
        if self.use_global_feat:
            self.global_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
            self.hand_global_attn = GlobalCrossAttention(
                self.inter_hidden, num_heads=self.global_cross_attn_heads
            )
            self.lip_global_attn = GlobalCrossAttention(
                self.inter_hidden, num_heads=self.global_cross_attn_heads
            )

        # Method 2: signer-invariant instance normalisation modules
        if self.use_feature_norm:
            self.signer_norm_spatial   = SignerNorm(self.inter_hidden)
            self.signer_norm_spatiotem = SignerNorm(self.inter_hidden)
            if self.use_global_feat:
                self.signer_norm_global = SignerNorm(self.inter_hidden)

        # if self.cross_modal_align:
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        # Phoneme tokenizer for CTC / Transformer alignment
        self.phoneme_tokenizer = PhonemeTokenizer()

        # CTC alignment module
        if self.align_mode == 'ctc':
            self.ctc_align = CTCAlign(
                input_dim=self.t5_model.config.hidden_size,
                vocab_size=self.phoneme_tokenizer.vocab_size,
                dropout=0.1,
            )

        # Transformer alignment module
        if self.align_mode == 'transformer':
            self.transformer_align = TransformerAlign(
                vocab_size=self.phoneme_tokenizer.vocab_size,
                input_dim=self.t5_model.config.hidden_size,
                d_model=self.align_decoder_dim,
                nhead=4,
                num_decoder_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                pad_id=self.phoneme_tokenizer.pad_id,
            )

        # Temporal attention pooling (replace global average pooling)
        if self.use_ta_pool:
            self.ta_pool = TemporalAttentionPool(self.t5_model.config.hidden_size)

    def prepare_inputs(
        self, 
        visual_outputs: torch.Tensor, 
        visual_mask: torch.Tensor, 
        samples: Dict, 
        split: str, 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Prepare combined inputs for the T5 model.
        
        Args:
            visual_outputs: Visual features
            visual_mask: Mask for visual features
            samples: Input samples
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (joint_outputs, joint_mask, output_tokens, targets)
        """
        bs = visual_outputs.shape[0]
        
        # Prepare the prompt with language information
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        # Tokenize prompts
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get lengths for visual and prompt sequences
        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths
        
        # Convert tokens to embeddings
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        # Concatenate visual and text embeddings
        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        # Pad the combined embeddings
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Prepare target labels (replace pad tokens with -100)
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        # Process spatial features if needed
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
            # Method 2: signer-invariant normalisation
            if self.use_feature_norm:
                spatial_outputs = self.signer_norm_spatial(spatial_outputs, spatial_mask)
#         Process spatiotemporal features if needed
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
            # Method 2: signer-invariant normalisation
            if self.use_feature_norm:
                spatiotemporal_outputs = self.signer_norm_spatiotem(
                    spatiotemporal_outputs, spatiotemporal_mask
                )

        # Process global features and apply cross-attention enrichment
        if self.use_global_feat and samples.get('global_values'):
            global_values = pad_sequence(samples['global_values'], batch_first=True)
            global_outputs = self.global_proj(global_values)
            global_mask = create_mask(seq_lengths=samples['global_lengths'], device=self.device)
            # Method 2: normalise global K/V distribution before cross-attention
            if self.use_feature_norm:
                global_outputs = self.signer_norm_global(global_outputs, global_mask)
            # Cross-attention: hand/lip Q attends to global K/V
            if spatial:
                spatial_outputs = self.hand_global_attn(
                    spatial_outputs, global_outputs, kv_mask=global_mask
                )
            if spatiotemporal:
                spatiotemporal_outputs = self.lip_global_attn(
                    spatiotemporal_outputs, global_outputs, kv_mask=global_mask
                )

        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length
            
            # Concatenate spatial and spatiotemporal features for each sample
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            
            # Apply temporal encoder
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            ) 
        else:
            # Use single feature type
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        Process batch inputs into a structured dictionary.
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Processed inputs dictionary
        """
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses, signers = [], [], []
        num_frames, glor_lengths, langs = [], [], []
        global_values, global_lengths = [], []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                # Calculate number of frames after sampling
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                # Collect metadata
                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample['gloss'])
                signers.append(sample.get('signer', ''))
                langs.append(sample['lang'])
                
                _ex_lang_trans = [
                    f"{sample['en_text']}={sample['text']}",
                    f"{sample['fr_text']}={sample['text']}",
                    f"{sample['es_text']}={sample['text']}"
                ]
                _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                ex_lang_translations.append(' '.join(_ex_lang_trans))
                
                # Handle too long sequences with random cropping
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                # Store processed visual features
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                # Process glor values if available
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))

                # Process global values if available
                gv = sample.get('global_value')
                if gv is not None and len(gv) > 0:
                    global_values.append(gv)
                    global_lengths.append(len(gv))

        # Only derange when in-context learning is active and translations are distinct.
        # When use_in_context=False all entries are empty strings, which would cause
        # derangement() to loop forever (can never satisfy x != x for identical values).
        if self.use_in_context and len(ex_lang_translations) > 1:
            ex_lang_translations = derangement(ex_lang_translations)

        # Return structured dictionary
        return {
            'pixel_values':   pixel_values,
            'glor_values':    glor_values,
            'global_values':  global_values,
            'bool_mask_pos':  masks,
            'ids':            ids,
            'text':           texts,
            'ex_lang_trans':  ex_lang_translations,
            'gloss':          glosses,
            'signers':        signers,
            'lang':           langs,
            'num_frames':     num_frames,
            'glor_lengths':   glor_lengths,
            'global_lengths': global_lengths,
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        """
        Calculate visual-textual alignment loss.

        Args:
            visual_outputs: Visual features (B, T, D), after fusion_proj
            visual_masks:   Bool mask (B, T), True for valid frames
            samples:        Input samples dict containing 'text'

        Returns:
            Contrastive loss
        """
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        # Get text embeddings from T5 encoder token embeddings
        text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)  # (B, L, D_t5)

        # --- Visual feature aggregation ---
        # visual_masks: (B, T), True = valid frame
        if self.use_ta_pool:
            # Single-head temporal attention pooling
            image_embeds = self.ta_pool(visual_outputs, visual_masks)  # (B, D_t5)
        else:
            # Global average pooling (original)
            v_mask = visual_masks.float().unsqueeze(-1)              # (B, T, 1)
            v_len  = v_mask.sum(1).clamp(min=1)                      # (B, 1)
            image_embeds = (visual_outputs * v_mask).sum(1) / v_len  # (B, D_t5)

        # --- Masked mean pooling for text features ---
        # attention_mask: (B, L), 1 = valid token (excludes padding)
        t_mask = output_tokens.attention_mask.float().unsqueeze(-1)  # (B, L, 1)
        t_len  = t_mask.sum(1).clamp(min=1)                          # (B, 1)
        text_embeds = (text_embeds * t_mask).sum(1) / t_len          # (B, D_t5)

        # Normalize features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Calculate cosine similarities with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        # Calculate contrastive loss
        loss = clip_loss(logits_per_text)
        
        return loss

    def ctc_align_loss(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        phoneme_tokens, phoneme_lengths = self.phoneme_tokenizer.collate_ctc(samples['text'])
        phoneme_tokens = phoneme_tokens.to(self.device)
        phoneme_lengths = phoneme_lengths.to(self.device)
        loss = self.ctc_align(visual_outputs, visual_masks, phoneme_tokens, phoneme_lengths)
        return loss

    def transformer_align_loss(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        tgt_tokens, _, tgt_mask = self.phoneme_tokenizer.collate(samples['text'])
        tgt_tokens = tgt_tokens.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        loss = self.transformer_align(visual_outputs, visual_masks, tgt_tokens, tgt_mask)
        return loss

    def compute_triplet_loss(
        self,
        embeddings: torch.Tensor,
        glosses: List[str],
        signers: List[str],
    ) -> Tuple[torch.Tensor, float]:
        """
        Multi-positive hard triplet loss（n 元损失）。

        当 batch 由 GKSampler 构造时（G 个 gloss × K 个 signer），每个 anchor 在
        batch 内有 K-1 个同 gloss 不同 signer 的正样本，构成 n=K 元损失：

            n 元 = anchor + (K-1) positives + hard negative
            loss = ReLU( max_pos_dist − min_neg_dist + margin )

        其中 max_pos_dist 取所有正样本中最难（最远）的，min_neg_dist 取所有负样本
        中最难（最近）的。当 batch 为普通随机 batch 时退化为标准 triplet loss。

        Multi-speaker (multiple unique signers):
            Positive : same gloss, different signer  (hard = farthest positive)
            Negative : different gloss, same signer  (hard = closest negative)
                       fallback → different gloss, any signer

        Single-speaker (only one unique signer):
            Positive : same gloss, different utterance
            Negative : different gloss, same speaker (hard = closest negative)

        Args:
            embeddings: (B, D) sentence-level embeddings (before L2-norm)
            glosses:    list of gloss strings, length B
            signers:    list of signer IDs, length B

        Returns:
            (loss, valid_ratio)  where valid_ratio = fraction of anchors with
            at least one valid positive *and* one valid negative.
        """
        B = embeddings.shape[0]
        emb = F.normalize(embeddings, dim=-1)   # (B, D)

        unique_signers = set(signers)
        is_single_speaker = len(unique_signers) == 1

        total_loss  = torch.tensor(0.0, device=emb.device)
        valid_count = 0

        for i in range(B):
            g_i, s_i = glosses[i], signers[i]

            if is_single_speaker:
                pos_idx = [j for j in range(B) if glosses[j] == g_i and j != i]
                neg_idx = [j for j in range(B) if glosses[j] != g_i]
            else:
                pos_idx = [j for j in range(B) if glosses[j] == g_i and signers[j] != s_i]
                neg_idx = [j for j in range(B) if glosses[j] != g_i and signers[j] == s_i]
                if not neg_idx:
                    neg_idx = [j for j in range(B) if glosses[j] != g_i]

            if not pos_idx or not neg_idx:
                continue

            anchor  = emb[i]
            pos_emb = emb[torch.tensor(pos_idx, device=emb.device)]  # (P, D)
            neg_emb = emb[torch.tensor(neg_idx, device=emb.device)]  # (N, D)

            # hard positive (farthest) + hard negative (closest) — n-tuple mining
            pos_dist = (anchor.unsqueeze(0) - pos_emb).norm(dim=-1).max()
            neg_dist = (anchor.unsqueeze(0) - neg_emb).norm(dim=-1).min()

            total_loss  += F.relu(pos_dist - neg_dist + self.triplet_margin)
            valid_count += 1

        loss        = total_loss / max(valid_count, 1)
        valid_ratio = valid_count / max(B, 1)
        return loss, valid_ratio

    def compute_supcon_loss(
        self,
        embeddings: torch.Tensor,
        glosses: List[str],
        signers: List[str],
    ) -> Tuple[torch.Tensor, float]:
        """
        Supervised Contrastive Loss（SupCon）。

        充分利用 GKSampler batch 内所有正样本对（同 gloss 不同 signer），
        而不只用最难的一对。梯度信号更稳定，尤其适合 K≥3 的场景。

            L_i = -1/|P(i)| * Σ_{p∈P(i)} log(
                      exp(z_i·z_p / τ) /
                      Σ_{a≠i} exp(z_i·z_a / τ)
                  )

        Args:
            embeddings: (B, D) sentence-level embeddings
            glosses:    list of gloss strings, length B
            signers:    list of signer IDs, length B

        Returns:
            (loss, valid_ratio)
        """
        B = embeddings.shape[0]
        emb = F.normalize(embeddings, dim=-1)   # (B, D)
        tau = self.supcon_temperature

        unique_signers = set(signers)
        is_single_speaker = len(unique_signers) == 1

        # cosine similarity matrix (B, B)
        sim = emb @ emb.T / tau

        # 对角线屏蔽（自身不参与）
        diag_mask = torch.eye(B, dtype=torch.bool, device=emb.device)

        total_loss  = torch.tensor(0.0, device=emb.device)
        valid_count = 0

        for i in range(B):
            g_i, s_i = glosses[i], signers[i]

            if is_single_speaker:
                pos_mask = torch.tensor(
                    [glosses[j] == g_i and j != i for j in range(B)],
                    dtype=torch.bool, device=emb.device
                )
            else:
                pos_mask = torch.tensor(
                    [glosses[j] == g_i and signers[j] != s_i for j in range(B)],
                    dtype=torch.bool, device=emb.device
                )

            if not pos_mask.any():
                continue

            # 分母：所有 j ≠ i 的样本
            denom_mask = ~diag_mask[i]                # (B,) True 表示 j≠i
            log_denom  = torch.logsumexp(sim[i][denom_mask], dim=0)

            # 分子：所有正样本
            log_sum_pos = sim[i][pos_mask].sum()      # sum of log-numerators

            n_pos = pos_mask.sum().item()
            total_loss  += -(log_sum_pos - n_pos * log_denom) / n_pos
            valid_count += 1

        loss        = total_loss / max(valid_count, 1)
        valid_ratio = valid_count / max(B, 1)
        return loss, valid_ratio

    def _compute_align_loss(self, visual_outputs, visual_masks, samples):
        if self.align_mode == 'ctc':
            return self.ctc_align_loss(visual_outputs, visual_masks, samples), 'ctc_loss'
        elif self.align_mode == 'transformer':
            return self.transformer_align_loss(visual_outputs, visual_masks, samples), 'transformer_loss'
        else:
            return self.visual_textual_align(visual_outputs, visual_masks, samples), 'contra_loss'

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Shared logic for training, validation and testing steps.
        
        Args:
            inputs: Input dictionary
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (loss, log_dict)
        """
        # Prepare visual inputs and project to match text embedding dimensions
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        # Keep 768-dim features for triplet loss (pooled before fusion_proj upsizes to T5 dim)
        visual_outputs_768 = visual_outputs
        visual_outputs = self.fusion_proj(visual_outputs)
        
        # Initialize logging dictionary
        log_dict = {}
        
        # STEP 1: Determine training mode and prepare inputs accordingly
        if self.cross_modal_align:
            # For pure contrastive learning or warm-up phase
            if self.warm_up_steps is None and not self.combined_loss:
                # Pure contrastive learning mode
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss, loss_key = self._compute_align_loss(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/{loss_key}"] = cont_loss
                loss = cont_loss
                
            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                # Warm-up phase with contrastive learning
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss, loss_key = self._compute_align_loss(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/{loss_key}"] = cont_loss
                loss = cont_loss
                
            else:
                # Combined loss mode (regular training + contrastive)
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )
                
                # Forward pass through T5 model
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                t5_loss = outputs.loss
                log_dict[f"{split}/loss"] = t5_loss
                
                # Add contrastive component if using combined loss
                cont_loss, loss_key = self._compute_align_loss(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                
                log_dict[f"{split}/{loss_key}"] = cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        else:
            # Standard training without contrastive learning
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Forward pass through T5 model
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        # STEP 2: Method 3 — cross-signer n-tuple / SupCon loss (training only)
        if (self.use_triplet_loss or self.use_supcon_loss) and split == 'train':
            # Masked mean-pool 768-dim features to get sentence embeddings
            vlen   = visual_masks.sum(1).float().unsqueeze(-1).clamp(min=1)  # (B, 1)
            pooled = (visual_outputs_768 * visual_masks.unsqueeze(-1)).sum(1) / vlen  # (B, 768)

            if self.use_supcon_loss:
                con_loss, valid_ratio = self.compute_supcon_loss(
                    pooled, inputs['gloss'], inputs['signers']
                )
                log_dict[f"{split}/supcon_loss"]         = con_loss
                log_dict[f"{split}/triplet_valid_ratio"] = torch.tensor(valid_ratio)
            else:
                con_loss, valid_ratio = self.compute_triplet_loss(
                    pooled, inputs['gloss'], inputs['signers']
                )
                log_dict[f"{split}/triplet_loss"]        = con_loss
                log_dict[f"{split}/triplet_valid_ratio"] = torch.tensor(valid_ratio)

            loss = loss + self.triplet_weight * con_loss

        # STEP 3: Handle evaluation phase (validation/testing)
        if split != "train":
            # Prepare inputs for text generation
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Generate translations
            generated = self.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=5,
                max_length=self.max_txt_len,
                top_p=0.9,
                do_sample=True,
            )
            
            # Decode generated outputs and references
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]
            
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)
            
            # Calculate evaluation metrics
            # eval_res = evaluate_results(
            #     predictions=generated_strings,
            #     references=reference_strings,
            #     split=split,
            #     tokenizer='zh' if inputs['lang'][0] == 'Chinese' else '13a',
            #     device=self.device
            # )
            
            # Add evaluation results to logging
            # log_dict.update(eval_res)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)
            
        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            tokenizer=self.eval_tokenizer,
            device=self.device
        )
        
        # Add evaluation results to logging
        # log_dict.update(eval_res)

        self.log_dict(eval_res, sync_dist=True)

        # Dictionary-based WER: maps phoneme sequences → Chinese via Pinyin1000 dictionary
        if self.dict_wer_evaluator is not None:
            dict_res = self.dict_wer_evaluator.compute_wer(
                self.generated, self.references, split='val'
            )
            print(f"[Dict WER] val/dict_wer={dict_res.get('val/dict_wer', 0):.2f}%  "
                  f"val/dict_cer={dict_res.get('val/dict_cer', 0):.2f}%  "
                  f"val/dict_ser={dict_res.get('val/dict_ser', 0):.2f}%")
            self.log_dict(dict_res, sync_dist=True)

        self.set_container()

    def on_test_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)

        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            tokenizer=self.eval_tokenizer,
            device=self.device
        )

        self.log_dict(eval_res, sync_dist=True)

        # Dictionary-based WER: maps phoneme sequences → Chinese via Pinyin1000 dictionary
        if self.dict_wer_evaluator is not None:
            dict_res = self.dict_wer_evaluator.compute_wer(
                self.generated, self.references, split='test'
            )
            print(f"[Dict WER] test/dict_wer={dict_res.get('test/dict_wer', 0):.2f}%  "
                  f"test/dict_cer={dict_res.get('test/dict_cer', 0):.2f}%  "
                  f"test/dict_ser={dict_res.get('test/dict_ser', 0):.2f}%")
            self.log_dict(dict_res, sync_dist=True)

        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            eps=1e-8, 
            weight_decay=0.01, 
            betas=(0.9, 0.98)
        )
        
        # Calculate total steps based on PyTorch Lightning trainer settings
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation if the attribute doesn't exist
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, 'dataloader'):
                train_dataloader = train_dataloader.dataloader
            
            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs
            
            # Account for gradient accumulation if used
            if hasattr(self.trainer, 'accumulate_grad_batches'):
                total_steps = total_steps // self.trainer.accumulate_grad_batches
        
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }