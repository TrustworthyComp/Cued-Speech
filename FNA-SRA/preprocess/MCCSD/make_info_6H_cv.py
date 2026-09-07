#!/usr/bin/env python3
"""
Generate LOSO (Leave-One-Signer-Out) cross-validation annotation files
for MCCSD 6-speaker (6H) dataset.

6 speakers: HS, XP, WT, LF, YX, YZ

For each fold (test_signer):
    train : other 5 signers, first (1-dev_ratio) portions by sentence index
    dev   : other 5 signers, last dev_ratio portions by sentence index
    test  : all data from the held-out signer

Input labels:
    multi_speaker_train_labels_6H.txt and multi_speaker_test_labels_6H.txt
    Format: signer,filename,gloss

Output files (per fold):
    fold_6H_{TEST}_train_info_ml.npy
    fold_6H_{TEST}_dev_info_ml.npy
    fold_6H_{TEST}_test_info_ml.npy

Usage:
    python preprocess/MCCSD/make_info_6H_cv.py \
        --data_root /home/uic2/mccsd_sub \
        --alt_roots /home/uic2/mccsd_datasets/RawVideo \
        --train_txt /home/uic2/mccsd_sub/multi_speaker_train_labels_6H.txt \
        --test_txt  /home/uic2/mccsd_sub/multi_speaker_test_labels_6H.txt \
        --save_dir  ./preprocess/MCCSD_6H_CV \
        --dev_ratio 0.1
"""

import argparse
import os
import os.path as osp
import re
import numpy as np


SIGNERS = ['HS', 'XP', 'WT', 'LF', 'YX', 'YZ']


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True,
                   help='Root directory of standard 4H video files')
    p.add_argument('--alt_roots', default='',
                   help='Comma-separated alternative roots for YX/YZ signers')
    p.add_argument('--train_txt', required=True,
                   help='multi_speaker_train_labels_6H.txt path')
    p.add_argument('--test_txt', required=True,
                   help='multi_speaker_test_labels_6H.txt path')
    p.add_argument('--save_dir', required=True,
                   help='Directory to write fold annotation npy files')
    p.add_argument('--signers', nargs='+', default=SIGNERS,
                   help='List of all signer IDs (default: HS XP WT LF YX YZ)')
    p.add_argument('--dev_ratio', type=float, default=0.1,
                   help='Fraction of training sentences to use as dev set')
    return p


def parse_label_line(line):
    """Parse a label line: signer,filename,gloss"""
    parts = line.strip().split(',')
    if len(parts) < 3:
        return None
    signer = parts[0]
    filename = parts[1]
    gloss = ','.join(parts[2:]).strip()
    fileid = filename.replace('.mp4', '')
    return signer, fileid, gloss


def load_all_entries(args):
    """Load all entries from train and test label files."""
    entries = []

    for label_path in [args.train_txt, args.test_txt]:
        if not osp.exists(label_path):
            print(f'[WARN] Label file not found: {label_path}')
            continue
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
                    'tag':                'mccsd_6h',
                    'en_text':            gloss,
                    'es_text':            gloss,
                    'fr_text':            gloss,
                    'lang':               'Chinese',
                    'phoneme_intervals':  [],
                    'duration_sec':       0.0,
                }
                entries.append(entry)

    return entries


def build_folds(entries, signers, dev_ratio, save_dir):
    """Build LOSO folds and save npy files."""
    signer_indices = {s: {} for s in signers}
    for i, entry in enumerate(entries):
        s = entry['signer']
        if s in signer_indices:
            fid = entry['fileid']
            signer_indices[s][i] = entry

    print(f'\nTotal entries loaded: {len(entries)}')
    for s in signers:
        n = len(signer_indices[s])
        print(f'  Signer {s}: {n} samples')

    for test_signer in signers:
        train_signers = [s for s in signers if s != test_signer]

        train_dict, dev_dict, test_dict = {}, {}, {}
        ti = di = xi = 0

        for s in signers:
            s_entries = list(signer_indices[s].items())
            s_entries.sort()

            if s == test_signer:
                for _, entry in s_entries:
                    test_dict[xi] = entry
                    xi += 1
            else:
                n = len(s_entries)
                n_dev = max(1, int(n * dev_ratio))
                n_train = n - n_dev

                for j, (idx, entry) in enumerate(s_entries):
                    if j < n_train:
                        train_dict[ti] = entry
                        ti += 1
                    else:
                        dev_dict[di] = entry
                        di += 1

        train_dict['num'] = len(train_dict)
        dev_dict['num'] = len(dev_dict)
        test_dict['num'] = len(test_dict)

        train_path = osp.join(save_dir, f'fold_6H_{test_signer}_train_info_ml.npy')
        dev_path   = osp.join(save_dir, f'fold_6H_{test_signer}_dev_info_ml.npy')
        test_path  = osp.join(save_dir, f'fold_6H_{test_signer}_test_info_ml.npy')

        np.save(train_path, train_dict)
        np.save(dev_path,   dev_dict)
        np.save(test_path,  test_dict)

        print(f'\nFold test={test_signer}  (train signers: {train_signers})')
        print(f'  Train: {len(train_dict)}  Dev: {len(dev_dict)}  Test: {len(test_dict)}')

    print(f'\nFiles saved to {save_dir}/')
    print('Done.')


def main():
    args = get_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    entries = load_all_entries(args)
    if not entries:
        print('[ERROR] No entries found. Check label file paths.')
        return

    build_folds(entries, args.signers, args.dev_ratio, args.save_dir)


if __name__ == '__main__':
    main()
