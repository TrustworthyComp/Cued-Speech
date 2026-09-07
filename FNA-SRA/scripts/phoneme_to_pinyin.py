#!/usr/bin/env python3
"""
Phoneme-to-Pinyin Converter using Dictionary Lookup

This script converts model output phoneme sequences to pinyin sequences
using a dictionary built from Pinyin1000.docx.

Usage:
    python scripts/phoneme_to_pinyin.py --help

Examples:
    # Convert single sequence
    python scripts/phoneme_to_pinyin.py --phoneme "w o - m en - q v - b y e"

    # Batch convert from file
    python scripts/phoneme_to_pinyin.py --input predictions.txt --output pinyin_output.txt

    # Build dictionary only
    python scripts/phoneme_to_pinyin.py --build-dict dict.json
"""

import argparse
import json
import os
import re
from docx import Document
from pathlib import Path


PHONEME_TO_PINYIN = {
    # Initials (声母)
    'b': 'b', 'p': 'p', 'm': 'm', 'f': 'f',
    'd': 'd', 't': 't', 'n': 'n', 'l': 'l',
    'g': 'g', 'k': 'k', 'h': 'h',
    'j': 'j', 'q': 'q', 'x': 'x',
    'zh': 'zh', 'ch': 'ch', 'sh': 'sh', 'r': 'r',
    'z': 'z', 'c': 'c', 's': 's',
    'y': 'y', 'w': 'w',

    # Finals (韵母) - single
    'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'v': 'v', 'er': 'er',

    # Finals - compound
    'ai': 'ai', 'ei': 'ei', 'ui': 'ui', 'ao': 'ao', 'ou': 'ou', 'iu': 'iu',
    'ie': 'ie', 've': 've', 'an': 'an', 'en': 'en', 'in': 'in', 'un': 'un', 'vn': 'vn',
    'ang': 'ang', 'eng': 'eng', 'ing': 'ing', 'ong': 'ong',

    # Special
    'n': 'n',   # for "n" as final after some initials
}


def parse_phoneme_sequence(phoneme_str):
    """Parse phoneme string like 'w o - m en - q v' into list of phonemes."""
    tokens = phoneme_str.replace('-', ' ').split()
    return tokens


def phonemes_to_pinyin(phoneme_list):
    """Convert list of phonemes to pinyin string."""
    result = []
    i = 0
    while i < len(phoneme_list):
        p = phoneme_list[i]

        if p == '-':
            i += 1
            continue

        # Check for zh, ch, sh
        if p in ['zh', 'ch', 'sh'] and i + 1 < len(phoneme_list):
            final = phoneme_list[i + 1]
            if final not in ['a', 'o', 'e', 'i', 'u', 'v', 'er',
                            'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
                            'an', 'en', 'in', 'un', 'vn',
                            'ang', 'eng', 'ing', 'ong']:
                result.append(p)
            else:
                result.append(p + final)
                i += 1
        # Check for special initials
        elif p in ['j', 'q', 'x'] and i + 1 < len(phoneme_list):
            final = phoneme_list[i + 1]
            if final == 'v':
                result.append(p + 'u')
                i += 1
            elif final in ['i', 'ia', 'ie', 'iao', 'iu', 'ian', 'iang', 'iong']:
                result.append(p + final)
                i += 1
            else:
                result.append(p)
        elif p in ['n', 'l'] and i + 1 < len(phoneme_list):
            final = phoneme_list[i + 1]
            if final == 'g':
                result.append(p + 'g')
                i += 1
            elif final in ['a', 'o', 'e', 'i', 'u', 'v', 'er',
                          'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
                          'an', 'en', 'in', 'un', 'vn',
                          'ang', 'eng', 'ing', 'ong']:
                result.append(p + final)
                i += 1
            else:
                result.append(p)
        # Check for regular initial + final
        elif i + 1 < len(phoneme_list):
            final = phoneme_list[i + 1]
            if final in ['a', 'o', 'e', 'i', 'u', 'v', 'er',
                        'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
                        'an', 'en', 'in', 'un', 'vn',
                        'ang', 'eng', 'ing', 'ong']:
                result.append(p + final)
                i += 1
            else:
                result.append(p)
        else:
            result.append(p)
        i += 1

    return ' '.join(result)


def load_pinyin_dict_from_docx(docx_path):
    """Load pinyin dictionary from Pinyin1000.docx."""
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    sentence_dict = {}
    i = 0
    while i < len(paras):
        line = paras[i]
        m = re.match(r'^(\d+）)\s*(.*)', line)
        if m:
            idx = int(m.group(1).replace('）', ''))
            chinese = m.group(2).strip()
            if i + 1 < len(paras):
                pinyin = paras[i + 1].strip()
                sentence_dict[idx] = {
                    'chinese': chinese,
                    'pinyin': pinyin
                }
            i += 2
        else:
            i += 1

    return sentence_dict


def build_phoneme_to_pinyin_dict(docx_path):
    """Build a mapping from phoneme pattern to pinyin."""
    sentence_dict = load_pinyin_dict_from_docx(docx_path)

    phoneme_to_pinyin = {}
    for idx, data in sentence_dict.items():
        pinyin = data['pinyin']
        phoneme_list = pinyin_to_phonemes(pinyin)
        phoneme_str = ' - '.join(phoneme_list)
        phoneme_to_pinyin[phoneme_str] = pinyin
        phoneme_to_pinyin[' '.join(phoneme_list)] = pinyin

    return phoneme_to_pinyin


def pinyin_to_phonemes(pinyin_str):
    """Convert pinyin string to phoneme list."""
    tokens = pinyin_str.replace('-', ' ').split()
    result = []

    for token in tokens:
        if not token or token in '，。！？':
            continue

        # Simple parsing - need more sophisticated approach for real use
        i = 0
        while i < len(token):
            matched = False
            # Try 3-char first
            if i + 3 <= len(token):
                for length in [3, 2, 1]:
                    sub = token[i:i+length]
                    if sub in PHONEME_TO_PINYIN:
                        result.append(sub)
                        i += length
                        matched = True
                        break
            else:
                for length in [2, 1]:
                    if i + length <= len(token):
                        sub = token[i:i+length]
                        if sub in PHONEME_TO_PINYIN:
                            result.append(sub)
                            i += length
                            matched = True
                            break
            if not matched:
                i += 1

    return result


def get_parser():
    p = argparse.ArgumentParser(description='Convert phoneme sequences to pinyin')
    p.add_argument('--phoneme', type=str, help='Single phoneme sequence')
    p.add_argument('--input', type=str, help='Input file with phoneme sequences')
    p.add_argument('--output', type=str, help='Output file for pinyin')
    p.add_argument('--build-dict', type=str, help='Save dictionary to JSON file')
    p.add_argument('--docx', default='/home/uic2/mccsd_datasets/Textfile/Pinyin1000.docx',
                   help='Path to Pinyin1000.docx')
    return p


def main():
    args = get_parser().parse_args()

    if args.build_dict:
        sentence_dict = load_pinyin_dict_from_docx(args.docx)
        with open(args.build_dict, 'w', encoding='utf-8') as f:
            json.dump(sentence_dict, f, ensure_ascii=False, indent=2)
        print(f"Dictionary saved to {args.build_dict}")
        return

    if args.phoneme:
        phoneme_list = parse_phoneme_sequence(args.phoneme)
        pinyin = phonemes_to_pinyin(phoneme_list)
        print(f"Phoneme: {args.phoneme}")
        print(f"Pinyin:  {pinyin}")
        return

    if args.input and args.output:
        with open(args.input, 'r', encoding='utf-8') as f:
            phonemes = [line.strip() for line in f if line.strip()]

        results = []
        for phoneme_str in phonemes:
            phoneme_list = parse_phoneme_sequence(phoneme_str)
            pinyin = phonemes_to_pinyin(phoneme_list)
            results.append(pinyin)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))

        print(f"Converted {len(results)} sequences")
        print(f"Output saved to {args.output}")
        return

    print("No action specified. Use --help for usage information.")


if __name__ == '__main__':
    main()
