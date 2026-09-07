#!/usr/bin/env python3
"""
Generate LOSO (Leave-One-Signer-Out) cross-validation annotation files
for the MHI-MCCSD (8-HI) dataset.

Signers are auto-discovered from the multi-speaker label files.

For each fold (test_signer):
    train : other signers, first (1-dev_ratio) portions by sentence index
    dev   : other signers, last dev_ratio portions by sentence index
    test  : all data from the held-out signer

Input labels:
    multi_speaker_train_labels.txt and multi_speaker_test_labels.txt
    Format: signer,filename,gloss

Output files (per fold):
    fold_mhi_{TEST}_train_info_ml.npy
    fold_mhi_{TEST}_dev_info_ml.npy
    fold_mhi_{TEST}_test_info_ml.npy

Usage:
    python preprocess/MHI_MCCSD/make_info_cv.py \
        --data_root  /home/uic2/mhi-mccsd \
        --save_dir   ./preprocess/MHI_MCCSD_CV \
        --dev_ratio 0.1
"""

import argparse
import os
import os.path as osp
import re
import numpy as np


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='/home/uic2/mhi-mccsd',
                   help='Root directory of MHI-MCCSD dataset')
    p.add_argument('--train_labels', default=None,
                   help='Path to multi_speaker_train_labels.txt '
                        '(default: {data_root}/multi_speaker_train_labels.txt)')
    p.add_argument('--test_labels', default=None,
                   help='Path to multi_speaker_test_labels.txt '
                        '(default: {data_root}/multi_speaker_test_labels.txt)')
    p.add_argument('--save_dir', default='./preprocess/MHI_MCCSD_CV',
                   help='Directory to write fold annotation npy files')
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
    return signer, filename, gloss


def load_all_entries(train_path, test_path):
    """Load all entries from train and test label files."""
    entries = []

    for label_path in [train_path, test_path]:
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
                signer, filename, gloss = parsed

                fileid = filename
                if fileid.lower().endswith('.mp4'):
                    fileid = fileid[:-4]
                fileid = re.sub(
                    r'^([A-Za-z]+\d*-)(\d+)$',
                    lambda m: m.group(1) + m.group(2).zfill(4),
                    fileid,
                )

                entry = {
                    'fileid':            fileid,
                    'folder':            f'{signer}/{fileid}',
                    'signer':            signer,
                    'gloss':             gloss.replace(' - ', ' ').strip(),
                    'text':              gloss.replace(' - ', ' ').strip(),
                    'num_frames':        -1,
                    'original_info':     f'{fileid}|{signer}',
                    'tag':               'mhi_mccsd',
                    'en_text':           gloss,
                    'es_text':           gloss,
                    'fr_text':           gloss,
                    'lang':              'Chinese',
                    'phoneme_intervals': [],
                    'duration_sec':      -1.0,
                }
                entries.append(entry)

    return entries


def build_folds(entries, dev_ratio, save_dir):
    """Build LOSO folds and save npy files."""
    signer_indices = {}
    for i, entry in enumerate(entries):
        s = entry['signer']
        if s not in signer_indices:
            signer_indices[s] = {}
        signer_indices[s][i] = entry

    signers = sorted(signer_indices.keys())

    print(f'\nTotal entries loaded: {len(entries)}')
    print(f'Discovered {len(signers)} signers: {signers}')
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

        train_path = osp.join(save_dir, f'fold_mhi_{test_signer}_train_info_ml.npy')
        dev_path   = osp.join(save_dir, f'fold_mhi_{test_signer}_dev_info_ml.npy')
        test_path  = osp.join(save_dir, f'fold_mhi_{test_signer}_test_info_ml.npy')

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

    train_path = args.train_labels or osp.join(args.data_root, 'multi_speaker_train_labels.txt')
    test_path  = args.test_labels or osp.join(args.data_root, 'multi_speaker_test_labels.txt')

    entries = load_all_entries(train_path, test_path)
    if not entries:
        print('[ERROR] No entries found. Check label file paths.')
        return

    build_folds(entries, args.dev_ratio, args.save_dir)


if __name__ == '__main__':
    main()
