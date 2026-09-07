#!/usr/bin/env python3
"""
Generate annotation files for MCCSD 6-speaker (6H) dataset.

Usage:
    python preprocess/MCCSD/make_info_6H.py \
        --data_root /home/uic2/mccsd_sub \
        --train_txt multi_speaker_train_labels_6H.txt \
        --test_txt multi_speaker_test_labels_6H.txt \
        --save_dir ./preprocess/MCCSD_6H
"""

import argparse
import os
import os.path as osp
import numpy as np
import re
import glob


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True,
                   help='Root directory containing YX/, YZ/ subdirs')
    p.add_argument('--alt_roots', default='',
                   help='Comma-separated alternative data roots (e.g., /path1,/path2) for other signers')
    p.add_argument('--train_txt', required=True,
                   help='Training labels .txt file (format: signer,filename,phonemes)')
    p.add_argument('--test_txt', required=True,
                   help='Test labels .txt file (format: signer,filename,phonemes)')
    p.add_argument('--save_dir', required=True,
                   help='Where to write {train,test}_info_ml.npy')
    return p


def parse_label_line(line):
    """Parse a line from multi_speaker_train_labels_6H.txt
    Format: signer,filename,gloss
    Example: HS,HS-0124.mp4,w o - m en - q v - b y e - d e - d i - f ang - b a
    """
    parts = line.strip().split(',')
    if len(parts) < 3:
        return None
    signer = parts[0]
    filename = parts[1]
    gloss = ','.join(parts[2:]).strip()

    fileid = filename.replace('.mp4', '')
    return {
        'signer': signer,
        'fileid': fileid,
        'gloss': gloss,
        'text': gloss,
    }


SIGNERS_WITH_VIDEO = {
    'YX', 'YZ',  # in --data_root
    'HS', 'XP', 'WT', 'LF',  # in --alt_roots
}


def build_entries(data_root, alt_roots, label_txt):
    """Build dataset entries from label file."""
    root_map = {}
    if alt_roots:
        alt_list = alt_roots.split(',')
        for alt in alt_list:
            if osp.isdir(alt):
                subdirs = [d for d in os.listdir(alt) if osp.isdir(osp.join(alt, d))]
                for d in subdirs:
                    root_map[d] = alt

    default_roots = [data_root]
    for signer in SIGNERS_WITH_VIDEO:
        if signer not in root_map:
            if osp.isdir(osp.join(data_root, signer)):
                root_map[signer] = data_root

    entries = []
    with open(label_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = parse_label_line(line)
            if entry is None:
                continue

            signer = entry['signer']
            fileid = entry['fileid']

            video_path = None
            if signer in root_map:
                candidate = osp.join(root_map[signer], signer, f'{fileid}.mp4')
                if osp.exists(candidate):
                    video_path = candidate

            if not video_path:
                for r in default_roots:
                    candidate = osp.join(r, signer, f'{fileid}.mp4')
                    if osp.exists(candidate):
                        video_path = candidate
                        break

            if not video_path:
                print(f'[WARN] Video not found: {signer}/{fileid}.mp4')
                continue

            phoneme_str = entry['gloss']
            tokens = phoneme_str.split(' - ') if phoneme_str else []
            phoneme_intervals = [[i, i+1, t] for i, t in enumerate(tokens)]

            full_entry = {
                'fileid': fileid,
                'folder': f'{signer}/{fileid}',
                'signer': signer,
                'gloss': entry['gloss'],
                'text': entry['text'],
                'num_frames': -1,
                'original_info': f'{fileid}|{signer}',
                'tag': 'mccsd_6h',
                'en_text': entry['text'],
                'es_text': entry['text'],
                'fr_text': entry['text'],
                'lang': 'Chinese',
                'phoneme_intervals': phoneme_intervals,
                'duration_sec': 0.0,
            }
            entries.append(full_entry)

    return entries


def main():
    args = get_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Building entries from {args.train_txt}...')
    train_entries = build_entries(args.data_root, args.alt_roots, args.train_txt)
    print(f'  Found {len(train_entries)} training samples')

    print(f'Building entries from {args.test_txt}...')
    test_entries = build_entries(args.data_root, args.alt_roots, args.test_txt)
    print(f'  Found {len(test_entries)} test samples')

    train_dict = {i: e for i, e in enumerate(train_entries)}
    train_dict['num'] = len(train_entries)
    train_path = osp.join(args.save_dir, 'train_info_ml.npy')
    np.save(train_path, train_dict)
    print(f'Saved {train_path}')

    test_dict = {i: e for i, e in enumerate(test_entries)}
    test_dict['num'] = len(test_entries)
    test_path = osp.join(args.save_dir, 'test_info_ml.npy')
    np.save(test_path, test_dict)
    print(f'Saved {test_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
