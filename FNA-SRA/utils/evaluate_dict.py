"""
Dictionary-based WER evaluation for Chinese Sign Language Recognition.

Strategy
--------
Instead of converting phoneme sequences to pinyin (which is fragile), we
build a lookup table directly from the dataset annotation files:

    flat_phoneme_string → Chinese sentence

The lookup is signer-aware: every signer has their own phoneme rendering of
the same 1 000 sentences, so we collect all known renderings.

For a given prediction phoneme string:
  1. Exact match in the lookup table → return the Chinese sentence directly.
  2. No exact match → find the nearest entry by *token-level* Levenshtein
     distance (each space-separated token = one unit).

Usage
-----
    from utils.evaluate_dict import DictionaryWER

    evaluator = DictionaryWER()
    metrics   = evaluator.compute_wer(predictions, references, split='test')
"""

import re
import os
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluate import _levenshtein


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_chinese(text: str) -> str:
    """Strip punctuation from a Chinese sentence."""
    return re.sub(r'[，。？！、；：""''…—\s]', '', text)


# ── main class ────────────────────────────────────────────────────────────────

class DictionaryWER:
    """Compute WER/CER by mapping phoneme sequences to Chinese via a lookup
    table built from the dataset annotation files."""

    def __init__(
        self,
        anno_glob: str = '/home/uic/fengling/mccsd/fna_sra/preprocess/MCCSD_6H*/*_info_ml.npy',
        docx_path: str = '/home/uic/fengling/mccsd/mccsd_datasets/Textfile/Pinyin1000.docx',
    ):
        self.phoneme_to_chinese: Dict[str, str] = {}   # flat phoneme → Chinese
        self._id_to_chinese:     Dict[int, str] = {}   # sentence id → Chinese

        self._load_dictionary(docx_path)
        self._build_lookup(anno_glob)

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_dictionary(self, docx_path: str):
        """Parse Pinyin1000.docx → {sentence_id: chinese_text}."""
        from docx import Document
        doc = Document(docx_path)
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        i = 0
        while i < len(paras):
            m = re.match(r'^(\d+)[）\)]\s*(.*)', paras[i])
            if m and i + 1 < len(paras):
                sid = int(m.group(1))
                chinese = m.group(2).strip()
                self._id_to_chinese[sid] = chinese
                i += 2
            else:
                i += 1
        print(f'[DictionaryWER] Loaded {len(self._id_to_chinese)} sentences from dictionary')

    def _build_lookup(self, anno_glob: str):
        """Build phoneme→Chinese lookup from all annotation npy files."""
        files = sorted(glob.glob(anno_glob))
        if not files:
            print(f'[DictionaryWER] WARNING: no annotation files found at {anno_glob}')
            return

        added = 0
        for fpath in files:
            try:
                data = np.load(fpath, allow_pickle=True).item()
            except Exception:
                continue
            for v in data.values():
                if not isinstance(v, dict):
                    continue
                fileid = v.get('fileid', '')
                nm = re.search(r'\d+', fileid)
                if not nm:
                    continue
                sid = int(nm.group())
                chinese = self._id_to_chinese.get(sid)
                if not chinese:
                    continue

                ivs = v.get('phoneme_intervals', [])
                flat = ' '.join(iv[2] for iv in ivs if iv[2]).strip()
                if flat and flat not in self.phoneme_to_chinese:
                    self.phoneme_to_chinese[flat] = chinese
                    added += 1

        print(f'[DictionaryWER] Built lookup: {added} phoneme→Chinese entries '
              f'from {len(files)} annotation files')

    # ── matching ──────────────────────────────────────────────────────────────

    def find_chinese(self, phoneme_str: str) -> Optional[str]:
        """Return the Chinese sentence for a phoneme string.

        Uses exact match first, then token-level nearest-neighbour.
        """
        phoneme_str = phoneme_str.strip()

        # 1. Exact match
        if phoneme_str in self.phoneme_to_chinese:
            return self.phoneme_to_chinese[phoneme_str]

        # 2. Token-level nearest-neighbour search
        pred_toks = phoneme_str.split()
        min_dist = float('inf')
        best_chinese = None

        for known_phoneme, chinese in self.phoneme_to_chinese.items():
            ref_toks = known_phoneme.split()
            dist = _levenshtein(pred_toks, ref_toks)
            if dist < min_dist:
                min_dist = dist
                best_chinese = chinese

        return best_chinese

    # ── metrics ───────────────────────────────────────────────────────────────

    def compute_cer_on_chinese(
        self,
        predictions: List[str],
        references:  List[str],
    ) -> float:
        """CER on Chinese characters after mapping both sides to Chinese."""
        total_dist = total_len = 0
        for pred_ph, ref_ph in zip(predictions, references):
            pred_zh = self.find_chinese(pred_ph)
            ref_zh  = self.find_chinese(ref_ph)
            if pred_zh and ref_zh:
                p = list(normalize_chinese(pred_zh))
                r = list(normalize_chinese(ref_zh))
                total_dist += _levenshtein(p, r)
                total_len  += len(r)
        return total_dist / max(total_len, 1) * 100

    def compute_sentence_er(
        self,
        predictions: List[str],
        references:  List[str],
    ) -> float:
        """Sentence Error Rate: fraction of sentences where mapped Chinese differs."""
        correct = 0
        for pred_ph, ref_ph in zip(predictions, references):
            pred_zh = self.find_chinese(pred_ph)
            ref_zh  = self.find_chinese(ref_ph)
            if pred_zh and ref_zh and normalize_chinese(pred_zh) == normalize_chinese(ref_zh):
                correct += 1
        return (1 - correct / max(len(predictions), 1)) * 100

    def compute_word_wer(
        self,
        predictions: List[str],
        references:  List[str],
    ) -> float:
        """True word-level WER using jieba segmentation on mapped Chinese."""
        import jieba
        jieba.setLogLevel(60)

        total_dist = total_len = 0
        for pred_ph, ref_ph in zip(predictions, references):
            pred_zh = self.find_chinese(pred_ph)
            ref_zh  = self.find_chinese(ref_ph)
            if pred_zh and ref_zh:
                pred_words = list(jieba.cut(normalize_chinese(pred_zh)))
                ref_words  = list(jieba.cut(normalize_chinese(ref_zh)))
                total_dist += _levenshtein(pred_words, ref_words)
                total_len  += len(ref_words)
        return total_dist / max(total_len, 1) * 100

    def compute_wer(
        self,
        predictions: List[str],
        references:  List[str],
        split:       str = 'val',
    ) -> Dict[str, float]:
        """Compute all WER metrics.

        Returns
        -------
        {split}/dict_wer  – word-level WER (jieba) on mapped Chinese
        {split}/dict_cer  – character-level CER on mapped Chinese
        {split}/dict_ser  – sentence error rate
        """
        if not predictions:
            return {f'{split}/dict_wer': 0.0,
                    f'{split}/dict_cer': 0.0,
                    f'{split}/dict_ser': 0.0}

        return {
            f'{split}/dict_wer': self.compute_word_wer(predictions, references),
            f'{split}/dict_cer': self.compute_cer_on_chinese(predictions, references),
            f'{split}/dict_ser': self.compute_sentence_er(predictions, references),
        }


# ── quick demo ────────────────────────────────────────────────────────────────

def demo():
    ev = DictionaryWER()

    preds = [
        'd ao sh i h ou w o q v b ang n i b an j y a',                  # correct
        'w o zh i ch i f a y an y eng g ai sh i x i j yu en g ang r an', # 1 token wrong
        'w o zh en d e h en ai t a',                                     # correct
    ]
    refs = [
        'd ao sh i h ou w o q v b ang n i b an j y a',
        'w o zh i ch i f a y an y eng g ai sh i x i j yu en g an r an',
        'w o zh en d e h en ai t a',
    ]

    print('\n=== Sample lookups ===')
    for p, r in zip(preds, refs):
        pc = ev.find_chinese(p)
        rc = ev.find_chinese(r)
        print(f'  pred → {pc}')
        print(f'  ref  → {rc}')
        print()

    print('=== Metrics ===')
    for k, v in ev.compute_wer(preds, refs, 'test').items():
        print(f'  {k}: {v:.2f}%')


if __name__ == '__main__':
    demo()
