from rouge_score import rouge_scorer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phoneme_to_pinyin_v3 import phoneme_to_syllables


# ---------------------------------------------------------------------------
# Character Error Rate / Word Error Rate for Chinese
# ---------------------------------------------------------------------------

def _levenshtein(seq_a: list, seq_b: list) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Character Error Rate for Chinese text.

    CER = (substitutions + deletions + insertions) / len(reference_chars) × 100

    Spaces are removed before comparison so that spacing differences in the
    model output do not inflate the error rate.
    """
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.replace(' ', ''))
        ref_chars  = list(ref.replace(' ', ''))
        total_dist += _levenshtein(pred_chars, ref_chars)
        total_len  += len(ref_chars)
    return total_dist / max(total_len, 1) * 100


def compute_per(predictions: list[str], references: list[str]) -> float:
    """
    Phoneme Error Rate (PER).

    Treats each whitespace-separated token as one phoneme unit and computes
    token-level edit distance.

    PER = (substitutions + deletions + insertions) / len(reference_phonemes) × 100
    """
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        pred_ph = pred.strip().split()
        ref_ph  = ref.strip().split()
        total_dist += _levenshtein(pred_ph, ref_ph)
        total_len  += len(ref_ph)
    return total_dist / max(total_len, 1) * 100


def compute_syllable_wer(predictions: list[str], references: list[str]) -> float:
    """
    Syllable-level WER for Chinese phoneme sequences.

    Each phoneme group that forms one Chinese character (one syllable) is
    treated as one 'word'.  The syllable parser in phoneme_to_pinyin_v3
    handles all medial (w/y/v/yu) combinations.

    Example
    -------
    Reference : i d i l y an l y an ...  →  [yi, di, lian, lian, ...]
    Generated : i zh i l y an l y an ... →  [yi, zhi, lian, lian, ...]
    edit_dist = 1  (di→zhi substitution, the standalone 'i' merged into 'yi' same)
    WER = 1 / len(reference_syllables)

    This is the canonical WER for Chinese SLT:
        one syllable = one Chinese character = one 'word'
    """
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        pred_syls = phoneme_to_syllables(pred)
        ref_syls  = phoneme_to_syllables(ref)
        total_dist += _levenshtein(pred_syls, ref_syls)
        total_len  += len(ref_syls)
    return total_dist / max(total_len, 1) * 100



    """
    Word Error Rate for Chinese text.

    Attempts to use jieba for word-level tokenisation.  Falls back to
    character-level tokenisation (identical to CER) if jieba is unavailable.

    WER = (substitutions + deletions + insertions) / len(reference_words) × 100
    """
    try:
        import jieba
        tokenise = lambda s: list(jieba.cut(s.replace(' ', '')))
    except ImportError:
        tokenise = lambda s: list(s.replace(' ', ''))

    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        pred_words = tokenise(pred)
        ref_words  = tokenise(ref)
        total_dist += _levenshtein(pred_words, ref_words)
        total_len  += len(ref_words)
    return total_dist / max(total_len, 1) * 100


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_results(
    predictions: list[str],
    references:  list[str],
    split:       str  = 'train',
    device:      str  = 'cpu',
    tokenizer:   str  = '13a',
) -> dict:
    """
    Evaluate predictions against references.

    tokenizer values:
      'zh'      – Chinese text: CER + WER (jieba)
      'phoneme' – Phoneme sequences: WER (token-level) + CER (char-level)
      '13a'     – English: WER + CER

    Args:
        predictions: Model outputs (list of strings).
        references:  Ground-truth strings.
        split:       'train', 'val', or 'test'.
        device:      Unused; kept for API compatibility.
        tokenizer:   Metric mode selector (see above).

    Returns:
        dict of metric_name → float value.
    """
    log_dicts = {}

    # ── CER + WER (primary metrics) ───────────────────────────────────────
    if split in ('val', 'test'):
        if tokenizer == 'phoneme':
            # PER: each individual phoneme token as one unit
            per = compute_per(predictions, references)
            log_dicts[f'{split}/per'] = per
            # Syllable WER: each syllable (= one Chinese character) as one word
            # This is the canonical WER for Chinese SLT
            wer = compute_syllable_wer(predictions, references)
            log_dicts[f'{split}/wer'] = wer
            # CER: character-level edit distance on the phoneme label strings
            cer = compute_cer(predictions, references)
            log_dicts[f'{split}/cer'] = cer
        else:
            cer = compute_cer(predictions, references)
            wer = compute_wer(predictions, references)
            log_dicts[f'{split}/cer'] = cer
            log_dicts[f'{split}/wer'] = wer

    # ── RougeL for test split (no BLEU) ───────────────────────────────────
    if split == 'test':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, pred)['rougeL']
            for ref, pred in zip(references, predictions)
        ]
        log_dicts[f'{split}/rougeL_f1'] = (
            sum(s.fmeasure for s in rouge_scores) / len(rouge_scores)
        )

    return log_dicts
