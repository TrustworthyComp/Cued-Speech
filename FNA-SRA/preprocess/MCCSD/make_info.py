# Preprocessing script: generate {train,dev,test}_info_ml.npy for MCCSD
#
# Dependencies:
#   pip install python-docx
#
# Usage:
#   cd /home/uic2/fengling/mccsd/FNA-SRA-main
#   python preprocess/MCCSD/make_info.py \
#       --anno_root   /home/uic2/mccsd_datasets/Annotation/video \
#       --text_root   /home/uic2/mccsd_datasets/Textfile \
#       --video_root  /home/uic2/mccsd_datasets/RawVideo \
#       --save_dir    ./preprocess/MCCSD
#
# Split strategy (adjustable via --train_end / --dev_end):
#   Sentences 0001-0800 → train   (3200 samples, all 4 signers)
#   Sentences 0801-0900 → dev     ( 400 samples)
#   Sentences 0901-1000 → test    ( 400 samples)

import argparse
import os
import os.path as osp
import glob
import re
import numpy as np
from docx import Document


SIGNERS = ['LF', 'HS', 'WT', 'XP']


# ---------------------------------------------------------------------------
# TextGrid parser
# ---------------------------------------------------------------------------

def parse_textgrid(tg_path: str):
    """
    Parse a Praat TextGrid file and return both the gloss string and
    the full interval list with timing.

    Returns:
        gloss_str    : space-joined non-empty labels  (str)
        intervals    : list of [t_start, t_end, label] for every non-empty
                       interval  (list[list])
        duration_sec : total audio duration (float)
    """
    intervals = []
    duration_sec = 0.0
    tokens = []

    with open(tg_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Parse duration from the global xmax line (first occurrence)
    for line in lines:
        m = re.match(r'\s*xmax\s*=\s*([\d.]+)', line)
        if m:
            duration_sec = float(m.group(1))
            break

    # Parse intervals: track current xmin/xmax, then text
    cur_xmin = cur_xmax = None
    in_interval = False
    for line in lines:
        line_s = line.strip()
        if line_s.startswith('intervals ['):
            in_interval = True
            cur_xmin = cur_xmax = None
            continue
        if in_interval:
            m_xmin = re.match(r'xmin\s*=\s*([\d.]+)', line_s)
            m_xmax = re.match(r'xmax\s*=\s*([\d.]+)', line_s)
            m_text = re.search(r'text\s*=\s*"([^"]*)"', line_s)
            if m_xmin:
                cur_xmin = float(m_xmin.group(1))
            elif m_xmax:
                cur_xmax = float(m_xmax.group(1))
            elif m_text:
                label = m_text.group(1).strip()
                if label and cur_xmin is not None and cur_xmax is not None:
                    intervals.append([cur_xmin, cur_xmax, label])
                    tokens.append(label)
                in_interval = False

    return ' '.join(tokens), intervals, duration_sec


# ---------------------------------------------------------------------------
# docx reader – reads sentences line by line from the document body
# ---------------------------------------------------------------------------

def parse_pinyin_docx(docx_path: str) -> tuple[list[str], list[str]]:
    """
    Parse Pinyin1000.docx which has this paired structure:

        1） 你好，很高兴认识你。          ← N）+ Chinese sentence
        ni hao ， hen gao xyeng ...      ← Pinyin on the very next line
        2） 你叫什么名字？
        ni jiao shen me myeng zi ？
        ...

    Header lines (title / category / subcategory) are skipped automatically
    because they don't match the "N）" pattern.

    Returns:
        (chinese_list, pinyin_list) – both lists are length 1000,
        index 0 = sentence 1.
    """
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    _is_header = re.compile(
        r'^(\d+\.|[一二三四五六七八九十]+[、，,]|汉语)'
    )

    chinese_map: dict[int, str] = {}
    pinyin_map:  dict[int, str] = {}

    i = 0
    while i < len(paras):
        line = paras[i]
        m = re.match(r'^(\d+)[）)]\s*(.*)', line)
        if m:
            idx     = int(m.group(1))
            chinese = m.group(2).strip()
            chinese_map[idx] = chinese
            # The line immediately after is the Pinyin (if it's not another sentence/header)
            if i + 1 < len(paras):
                nxt = paras[i + 1]
                if not re.match(r'^\d+[）)]', nxt) and not _is_header.match(nxt):
                    pinyin_map[idx] = nxt
                    i += 2
                    continue
        i += 1

    max_idx = max(chinese_map.keys()) if chinese_map else 0
    chinese_list = [chinese_map.get(j, '') for j in range(1, max_idx + 1)]
    pinyin_list  = [pinyin_map.get(j,  '') for j in range(1, max_idx + 1)]
    return chinese_list, pinyin_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--anno_root',  required=True,
                   help='Path to Annotation/video (contains LF/, HS/, ...)')
    p.add_argument('--text_root',  required=True,
                   help='Path to Textfile/ (contains Pinyin1000.docx)')
    p.add_argument('--video_root', default=None,
                   help='Path to RawVideo/ (reserved, not used currently)')
    p.add_argument('--save_dir',   required=True,
                   help='Where to write annotation .npy files')
    p.add_argument('--signers', nargs='+', default=SIGNERS)
    # Cross-validation mode (default)
    p.add_argument('--dev_ratio',  type=float, default=0.1,
                   help='Fraction of training sentences to use as dev set (default 0.1 = last 10%%)')
    # Hold-out mode
    p.add_argument('--split_mode', default='loso',
                   choices=['loso', 'holdout'],
                   help='loso: leave-one-speaker-out CV (default); '
                        'holdout: sentence-level 80/20 split across all signers')
    p.add_argument('--train_ratio', type=float, default=0.8,
                   help='Fraction of sentences for training in holdout mode (default 0.8)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for holdout sentence shuffle (default 42)')
    return p


# ---------------------------------------------------------------------------
# Entry-building helper
# ---------------------------------------------------------------------------

def _build_entry(signer, idx, chinese_sentences, pinyin_sentences, tg_path):
    fileid = f'{signer}-{idx:04d}'

    # Always parse TextGrid to get timing info
    tg_gloss, phoneme_intervals, duration_sec = parse_textgrid(tg_path)

    gloss = (pinyin_sentences[idx - 1]
             if idx - 1 < len(pinyin_sentences) and pinyin_sentences[idx - 1]
             else tg_gloss)
    text  = (chinese_sentences[idx - 1]
             if idx - 1 < len(chinese_sentences) and chinese_sentences[idx - 1]
             else '')
    return {
        'fileid':             fileid,
        'folder':             f'{signer}/{fileid}',
        'signer':             signer,
        'gloss':              gloss,
        'text':               text,
        'num_frames':         -1,
        'original_info':      f'{fileid}|{signer}|{idx}',
        'tag':                'mccsd',
        'en_text':            text,
        'es_text':            text,
        'fr_text':            text,
        'lang':               'Chinese',
        # ── TextGrid timing fields ──────────────────────────────────────
        # phoneme_intervals: [[t_start, t_end, label], ...]
        # Used by dataset.mccsd.MCCSD for phoneme-segment feature pooling
        # and by the CTC auxiliary loss.
        'phoneme_intervals':  phoneme_intervals,
        'duration_sec':       duration_sec,
    }


def _save(data_dict, path):
    data_dict['num'] = len(data_dict)
    np.save(path, data_dict)
    print(f'  Saved {path}  ({len(data_dict) - 1} samples)')


# ---------------------------------------------------------------------------
# 4-fold leave-one-speaker-out cross-validation
# ---------------------------------------------------------------------------

def make_cv_folds(args, chinese_sentences, pinyin_sentences):
    """
    4-fold LOSO cross-validation split.

    For each fold (test_signer ∈ {LF, HS, WT, XP}):
      train  : other 3 signers × sentences 1  … (1000 × (1-dev_ratio))
      dev    : other 3 signers × sentences (1000 × (1-dev_ratio)+1) … 1000
      test   : test_signer    × sentences 1  … 1000   (all 1000)

    Files written:
      fold_{TEST}_train_info_ml.npy
      fold_{TEST}_dev_info_ml.npy
      fold_{TEST}_test_info_ml.npy
    """
    # Preload all entries keyed by (signer, idx)
    all_entries: dict[tuple, dict] = {}
    for signer in args.signers:
        tg_files = sorted(glob.glob(
            osp.join(args.anno_root, signer, f'{signer}-*-V.TextGrid')
        ))
        for tg_path in tg_files:
            m = re.match(r'([A-Z]+)-(\d+)-V\.TextGrid', osp.basename(tg_path))
            if not m:
                continue
            idx = int(m.group(2))
            all_entries[(signer, idx)] = _build_entry(
                signer, idx, chinese_sentences, pinyin_sentences, tg_path
            )

    total_sentences = max(idx for (_, idx) in all_entries.keys())
    dev_start = int(total_sentences * (1.0 - args.dev_ratio)) + 1  # e.g. 901

    print(f'\nCV split: train sentences 1–{dev_start-1}, '
          f'dev sentences {dev_start}–{total_sentences}, '
          f'test = all {total_sentences} sentences of held-out signer\n')

    for test_signer in args.signers:
        train_signers = [s for s in args.signers if s != test_signer]

        train_dict, dev_dict, test_dict = {}, {}, {}
        ti = di = xi = 0

        for (signer, idx), entry in sorted(all_entries.items()):
            if signer == test_signer:
                test_dict[xi] = entry
                xi += 1
            else:
                if idx < dev_start:
                    train_dict[ti] = entry
                    ti += 1
                else:
                    dev_dict[di] = entry
                    di += 1

        print(f'Fold test={test_signer}  '
              f'(train signers: {train_signers})')
        _save(train_dict, osp.join(args.save_dir,
                                   f'fold_{test_signer}_train_info_ml.npy'))
        _save(dev_dict,   osp.join(args.save_dir,
                                   f'fold_{test_signer}_dev_info_ml.npy'))
        _save(test_dict,  osp.join(args.save_dir,
                                   f'fold_{test_signer}_test_info_ml.npy'))


# ---------------------------------------------------------------------------
# Sentence-level hold-out split (80 % train / 20 % test, all signers in both)
# ---------------------------------------------------------------------------

def make_holdout_split(args, chinese_sentences, pinyin_sentences):
    """
    Hold-out split: sentences are randomly divided 80 % / 20 %; every signer
    contributes samples to *both* train and test.

    Split principle
    ---------------
    - All sentence indices are shuffled with a fixed seed (reproducible).
    - The first (train_ratio × N) sentences go to train; the rest to test.
    - The same sentence partition is applied identically to every signer so
      that there is zero sentence overlap between train and test.
    - There is no separate dev set; the framework monitors the test set during
      training (holdout_test_info_ml.npy is used for both validation and test).

    Files written
    -------------
      holdout_train_info_ml.npy  — 80 % × |signers| samples
      holdout_test_info_ml.npy   — 20 % × |signers| samples  (val + test)
    """
    import random as _random
    _random.seed(args.seed)

    # ── Preload all entries ─────────────────────────────────────────────────
    all_entries: dict[tuple, dict] = {}
    for signer in args.signers:
        tg_files = sorted(glob.glob(
            osp.join(args.anno_root, signer, f'{signer}-*-V.TextGrid')
        ))
        for tg_path in tg_files:
            m = re.match(r'([A-Z]+)-(\d+)-V\.TextGrid', osp.basename(tg_path))
            if not m:
                continue
            idx = int(m.group(2))
            all_entries[(signer, idx)] = _build_entry(
                signer, idx, chinese_sentences, pinyin_sentences, tg_path
            )

    # ── Sentence-level random split ─────────────────────────────────────────
    all_sentence_ids = sorted({idx for (_, idx) in all_entries.keys()})
    shuffled = all_sentence_ids[:]
    _random.shuffle(shuffled)

    n_train = int(len(shuffled) * args.train_ratio)
    train_ids = set(shuffled[:n_train])
    test_ids  = set(shuffled[n_train:])

    print(f'\nHold-out split (seed={args.seed}):')
    print(f'  Train sentences: {n_train}  ({len(train_ids)} IDs)')
    print(f'  Test  sentences: {len(shuffled) - n_train}  ({len(test_ids)} IDs)')
    print(f'  Signers: {args.signers}')

    train_dict, test_dict = {}, {}
    ti = xi = 0

    for (signer, idx), entry in sorted(all_entries.items()):
        if idx in train_ids:
            train_dict[ti] = entry
            ti += 1
        else:
            test_dict[xi] = entry
            xi += 1

    _save(train_dict, osp.join(args.save_dir, 'holdout_train_info_ml.npy'))
    _save(test_dict,  osp.join(args.save_dir, 'holdout_test_info_ml.npy'))
    print('  (holdout_test_info_ml.npy is used for both validation and test)\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    pinyin_docx = osp.join(args.text_root, 'Pinyin1000.docx')
    if osp.exists(pinyin_docx):
        chinese_sentences, pinyin_sentences = parse_pinyin_docx(pinyin_docx)
        print(f'Loaded {len(chinese_sentences)} Chinese + '
              f'{len(pinyin_sentences)} Pinyin sentences.')
    else:
        print(f'[WARN] {pinyin_docx} not found – text/gloss will be empty.')
        chinese_sentences = [''] * 1000
        pinyin_sentences  = [''] * 1000

    if args.split_mode == 'holdout':
        make_holdout_split(args, chinese_sentences, pinyin_sentences)
    else:
        make_cv_folds(args, chinese_sentences, pinyin_sentences)
    print('\nDone.')


if __name__ == '__main__':
    main()
