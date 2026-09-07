#!/usr/bin/env python3
"""
Extract single-speaker subsets from the 6-speaker MCCSD dataset.

This script extracts training and test data for a specific speaker from the
6-speaker dataset (6H) with 4:1 split already applied.

Usage:
    python preprocess/MCCSD/extract_single_speaker.py --speaker HS --anno_root ./preprocess/MCCSD_6H --save_dir ./preprocess/MCCSD_6H_HS
"""

import argparse
import os
import os.path as osp
import numpy as np


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--speaker', required=True,
                   help='Speaker ID (e.g., HS, XP, WT, LF, YX, YZ)')
    p.add_argument('--anno_root', required=True,
                   help='Root directory containing train_info_ml.npy and test_info_ml.npy')
    p.add_argument('--save_dir', required=True,
                   help='Where to write {train,test}_info_ml.npy for this speaker')
    return p


def main():
    args = get_parser().parse_args()

    speaker = args.speaker.upper()
    print(f'Extracting data for speaker: {speaker}')

    train_path = osp.join(args.anno_root, 'train_info_ml.npy')
    test_path = osp.join(args.anno_root, 'test_info_ml.npy')

    print(f'Loading {train_path}...')
    train_data = np.load(train_path, allow_pickle=True).item()
    print(f'Loading {test_path}...')
    test_data = np.load(test_path, allow_pickle=True).item()

    train_dict = {}
    test_dict = {}

    ti = xi = 0
    for k, v in train_data.items():
        if k == 'num':
            continue
        if v.get('signer', '').upper() == speaker:
            train_dict[ti] = v
            ti += 1

    for k, v in test_data.items():
        if k == 'num':
            continue
        if v.get('signer', '').upper() == speaker:
            test_dict[xi] = v
            xi += 1

    print(f'Found {ti} training samples, {xi} test samples for speaker {speaker}')

    os.makedirs(args.save_dir, exist_ok=True)

    train_dict['num'] = ti
    train_out = osp.join(args.save_dir, 'train_info_ml.npy')
    np.save(train_out, train_dict)
    print(f'Saved {train_out}')

    test_dict['num'] = xi
    test_out = osp.join(args.save_dir, 'test_info_ml.npy')
    np.save(test_out, test_dict)
    print(f'Saved {test_out}')

    print('Done.')


if __name__ == '__main__':
    main()
