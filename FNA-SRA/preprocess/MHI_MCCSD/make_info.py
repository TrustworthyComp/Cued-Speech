"""
Preprocessing script: generate train_info_ml.npy and test_info_ml.npy
for the MHI-MCCSD dataset.

Label file format (multi_speaker_train/test_labels.txt):
    signer,filename.mp4,phoneme_label
    e.g.: F001,F001-0530.mp4,n y en - k an - sh i - u - w an - d e - b ao - e - g ou - b u - g ou

The phoneme label uses ' - ' as syllable separator. This script flattens
all phonemes into a single space-separated string for the 'gloss' field.

Usage:
    cd /home/uic2/fengling/mccsd/FNA-SRA-main
    python preprocess/MHI_MCCSD/make_info.py \
        --data_root  /home/uic2/mhi-mccsd \
        --save_dir   preprocess/MHI_MCCSD

Output files:
    {save_dir}/train_info_ml.npy
    {save_dir}/test_info_ml.npy
"""

import argparse
import os
import numpy as np


def parse_label_file(label_path: str) -> list[dict]:
    """
    Parse a MHI-MCCSD label file and return a list of entry dicts.

    Each line: signer,filename.mp4,phoneme_label
    Returns list of dicts with fields compatible with MCCSD dataset class.
    """
    entries = []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 2)
            if len(parts) != 3:
                print(f'[WARN] Skipping malformed line: {line!r}')
                continue
            signer, filename, phoneme_label = parts

            # fileid: strip .mp4 extension and normalise number to 4-digit zero-padded
            # e.g. "F002-738.mp4" → "F002-0738" to match feature filenames on disk
            fileid = filename
            if fileid.lower().endswith('.mp4'):
                fileid = fileid[:-4]
            import re as _re
            fileid = _re.sub(
                r'^([A-Za-z]+\d*-)(\d+)$',
                lambda m: m.group(1) + m.group(2).zfill(4),
                fileid,
            )

            # Gloss: replace syllable separator ' - ' with ' ' to get flat
            # space-separated phoneme sequence (e.g. "n y en k an sh i ...")
            gloss = phoneme_label.replace(' - ', ' ').strip()

            entry = {
                'fileid':            fileid,
                'folder':            f'{signer}/{fileid}',
                'signer':            signer,
                'gloss':             gloss,
                'text':              gloss,   # no Chinese text available
                'num_frames':        -1,
                'original_info':     f'{fileid}|{signer}',
                'tag':               'mhi_mccsd',
                'en_text':           gloss,
                'es_text':           gloss,
                'fr_text':           gloss,
                'lang':              'Chinese',
                # No TextGrid timing available; dataset falls back to gloss
                'phoneme_intervals': [],
                'duration_sec':      -1.0,
            }
            entries.append(entry)
    return entries


def save_npy(entries: list[dict], path: str) -> None:
    data = {i: e for i, e in enumerate(entries)}
    data['num'] = len(entries)
    np.save(path, data)
    print(f'  Saved {path}  ({len(entries)} samples)')


def main():
    parser = argparse.ArgumentParser(
        description='Build MHI-MCCSD annotation npy files for FNA-SRA framework.'
    )
    parser.add_argument('--data_root', default='/home/uic2/mhi-mccsd',
                        help='Root directory of MHI-MCCSD dataset')
    parser.add_argument('--train_labels', default=None,
                        help='Path to train labels file '
                             '(default: {data_root}/multi_speaker_train_labels.txt)')
    parser.add_argument('--test_labels', default=None,
                        help='Path to test labels file '
                             '(default: {data_root}/multi_speaker_test_labels.txt)')
    parser.add_argument('--save_dir', default='preprocess/MHI_MCCSD',
                        help='Directory to write annotation npy files')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_path = args.train_labels or os.path.join(
        args.data_root, 'multi_speaker_train_labels.txt'
    )
    test_path = args.test_labels or os.path.join(
        args.data_root, 'multi_speaker_test_labels.txt'
    )

    print(f'Reading train labels: {train_path}')
    train_entries = parse_label_file(train_path)

    print(f'Reading test labels:  {test_path}')
    test_entries = parse_label_file(test_path)

    save_npy(train_entries, os.path.join(args.save_dir, 'train_info_ml.npy'))
    save_npy(test_entries,  os.path.join(args.save_dir, 'test_info_ml.npy'))

    print('\nDone.')
    print(f'  Train: {len(train_entries)} samples')
    print(f'  Test:  {len(test_entries)} samples')
    print(f'\nUsage example in your config / training script:')
    print(f'  anno_root      = "{os.path.abspath(args.save_dir)}"')
    print(f'  hand_feat_root = "/home/uic2/mhi-mccsd/Features/clip-vit-large-patch14_hand_feat_mccsd"')
    print(f'  lip_feat_root  = "/home/uic2/mhi-mccsd/Features/clip-vit-large-patch14_lip_feat_mccsd"')


if __name__ == '__main__':
    main()
