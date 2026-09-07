import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fna_sra.decoder_utils import PhonemeTokenizer


class CTCAlign(nn.Module):
    """
    CTC alignment module for phoneme-level supervision.

    Projects visual features to phoneme vocabulary logits and computes
    CTC loss against the target phoneme sequence.  Acts as a drop-in
    replacement for the CLIP-style contrastive alignment loss.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.ctc_proj = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(
        self,
        visual_features: torch.Tensor,
        visual_mask: torch.Tensor,
        phoneme_tokens: torch.Tensor,
        phoneme_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, D) after temporal encoding + projection
            visual_mask:     (B, T) bool, True = valid frame
            phoneme_tokens:  (B, L) padded token IDs (without blank/sos/eos)
            phoneme_lengths: (B,) actual lengths of each phoneme sequence

        Returns:
            CTC loss (scalar)
        """
        logits = self.ctc_proj(self.dropout(visual_features))  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)              # (B, T, V)

        input_lengths = visual_mask.sum(1).long()              # (B,)

        log_probs_t = log_probs.transpose(0, 1)               # (T, B, V)

        loss = self.ctc_loss(
            log_probs_t,
            phoneme_tokens,
            input_lengths,
            phoneme_lengths,
        )
        return loss

    def decode(self, visual_features: torch.Tensor, visual_mask: torch.Tensor) -> list:
        logits = self.ctc_proj(visual_features)                # (B, T, V)
        argmax_ids = torch.argmax(logits, dim=-1)              # (B, T)

        results = []
        for b in range(argmax_ids.size(0)):
            length = visual_mask[b].sum().item()
            ids_b = argmax_ids[b, :length].tolist()
            collapsed = []
            prev = -1
            for tid in ids_b:
                if tid != prev and tid != 0:
                    collapsed.append(tid)
                prev = tid
            results.append(collapsed)
        return results
