import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerAlign(nn.Module):
    """
    Transformer-decoder-based alignment module for phoneme-level supervision.

    Uses a standard Transformer decoder that cross-attends to visual features
    (as memory) and autoregressively predicts phoneme tokens.  Replaces the
    CLIP-style contrastive alignment loss with a standard CE loss.
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 2048,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.encoder_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    def forward(
        self,
        visual_features: torch.Tensor,
        visual_mask: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T_enc, D_enc) encoder features from visual stream
            visual_mask:     (B, T_enc) bool, True = valid frame
            tgt_tokens:      (B, L) padded target token IDs (including sos/eos)
            tgt_mask:        (B, L) bool, True = valid token

        Returns:
            CE loss (scalar)
        """
        memory = self.encoder_proj(visual_features)

        tgt_emb = self.tok_embed(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        tgt_seq_len = tgt_tokens.size(1)
        tgt_causal_mask = _generate_square_subsequent_mask(tgt_seq_len, tgt_emb.device)
        tgt_causal_mask = tgt_causal_mask.to(dtype=tgt_emb.dtype)

        memory_key_padding_mask = ~visual_mask

        tgt_key_padding_mask = ~tgt_mask

        decoder_output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_proj(decoder_output)                           # (B, L, V)

        shift_logits = logits[:, :-1, :].contiguous()                       # (B, L-1, V)
        shift_labels = tgt_tokens[:, 1:].contiguous()                       # (B, L-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.pad_id,
        )
        return loss

    def generate(
        self,
        visual_features: torch.Tensor,
        visual_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding from visual features.

        Returns:
            (B, L) token IDs
        """
        B = visual_features.size(0)
        device = visual_features.device

        memory = self.encoder_proj(visual_features)
        memory_key_padding_mask = ~visual_mask

        tgt_ids = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = self.tok_embed(tgt_ids) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            sz = tgt_ids.size(1)
            tgt_causal_mask = _generate_square_subsequent_mask(sz, device)
            tgt_causal_mask = tgt_causal_mask.to(dtype=tgt_emb.dtype)

            out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            logits = self.output_proj(out[:, -1:, :])   # (B, 1, V)
            next_token = logits.argmax(dim=-1)           # (B, 1)
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return tgt_ids
