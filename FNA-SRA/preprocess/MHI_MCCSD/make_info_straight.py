#!/usr/bin/env python3
"""
Generate straight train / dev / test annotation files for MHI-MCCSD 8H
from pre-defined label splits (no speaker-based cross-validation).

Reads:
    multi_speaker_train_labels.txt   →  train (90%) + dev (10%)
    multi_speaker_test_labels.txt    →  test  (100%)

Output files:
    {save_dir}/train_info_ml.npy
    {save_dir}/dev_info_ml.npy
    {save_dir}/test_info_ml.npy

Usage:
    python preprocess/MHI_MCCSD/make_info_straight.py \
        --train_txt  /home/uic/fengling/mccsd/mhi-mccsd/multi_speaker_train_labels.txt \
        --test_txt   /home/uic/fengling/mccsd/mhi-mccsd/multi_speaker_test_labels.txt \
        --save_dir   ./preprocess/MHI_MCCSD_8H_STRAIGHT \
        --dev_ratio  0.1
"""

import argparse
import os
import os.path as osp
import numpy as np
import re as _re


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--train_txt', required=True)
    p.add_argument('--test_txt', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--dev_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    return p


def parse_label_line(line):
    parts = line.strip().split(',', 2)
    if len(parts) < 3:
        return None
    signer = parts[0]
    filename = parts[1]
    phoneme_label = parts[2].strip()

    # fileid: strip .mp4 extension and zero-pad numeric suffix (MHI convention)
    fileid = filename
    if fileid.lower().endswith('.mp4'):
        fileid = fileid[:-4]
    fileid = _re.sub(
        r'^([A-Za-z]+\d*-)(\d+)$',
        lambda m: m.group(1) + m.group(2).zfill(4),
        fileid,
    )

    # Gloss: replace syllable separator ' - ' with ' ' (flat phoneme sequence)
    gloss = phoneme_label.replace(' - ', ' ').strip()

    return signer, fileid, gloss


def load_entries(label_path):
    entries = []
    if not osp.exists(label_path):
        print(f'[WARN] Label file not found: {label_path}')
        return entries
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            signer, fileid, gloss = parsed
            entry = {
                'fileid':             fileid,
                'folder':             f'{signer}/{fileid}',
                'signer':             signer,
                'gloss':              gloss,
                'text':               gloss,
                'num_frames':         -1,
                'original_info':      f'{fileid}|{signer}',
                'tag':                'mhi_mccsd_8h',
                'en_text':            gloss,
                'es_text':            gloss,
                'fr_text':            gloss,
                'lang':               'Chinese',
                'phoneme_intervals':  [],
                'duration_sec':       -1.0,
            }
            entries.append(entry)
    return entries


def main():
    args = get_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_entries = load_entries(args.train_txt)
    test_entries = load_entries(args.test_txt)

    import random
    rng = random.Random(args.seed)
    rng.shuffle(train_entries)

    n_dev = max(1, int(len(train_entries) * args.dev_ratio))
    dev_entries = train_entries[:n_dev]
    train_entries = train_entries[n_dev:]

    print(f'Train entries: {len(train_entries)}')
    print(f'Dev   entries: {len(dev_entries)}')
    print(f'Test  entries: {len(test_entries)}')

    for name, entries in [('train', train_entries), ('dev', dev_entries), ('test', test_entries)]:
        data_dict = {}
        for i, entry in enumerate(entries):
            data_dict[i] = entry
        data_dict['num'] = len(data_dict)
        out_path = osp.join(args.save_dir, f'{name}_info_ml.npy')
        np.save(out_path, data_dict)
        print(f'  {name}_info_ml.npy  →  {len(entries)} samples')

    print(f'\nFiles saved to {args.save_dir}/')
    print('Done.')


if __name__ == '__main__':
    main()
