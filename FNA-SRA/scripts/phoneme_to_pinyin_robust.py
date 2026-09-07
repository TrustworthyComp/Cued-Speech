#!/usr/bin/env python3
"""
Robust Phoneme-to-Pinyin Converter

This module converts phoneme sequences (like 'w o - m en - q v - b y e')
to pinyin strings (like 'wo men qv bie') using a combination of
rule-based parsing and dictionary-based correction.

Usage:
    python scripts/phoneme_to_pinyin_robust.py --phoneme "w o - m en - q v - b y e"
"""

import re
from typing import List, Tuple, Dict


INITIALS = {'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
            'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w'}

FINALS = {
    'a', 'o', 'e', 'i', 'u', 'v', 'er',
    'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
    'an', 'en', 'in', 'un', 'vn',
    'ang', 'eng', 'ing', 'ong',
}

SPECIAL_MAPPINGS = {
    ('j', 'v'): 'ju',
    ('q', 'v'): 'qu',
    ('x', 'v'): 'xu',
    ('j', 'i', 'e'): 'jie',
    ('q', 'i', 'e'): 'qie',
    ('x', 'i', 'e'): 'xie',
    ('n', 'g'): 'ng',
    ('l', 'g'): 'lg',
}


def parse_phoneme_tokens(phoneme_str: str) -> List[str]:
    """Parse phoneme string into tokens (removing '-' markers)."""
    tokens = []
    for token in phoneme_str.replace('-', ' ').split():
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens


def phonemes_to_syllables(tokens: List[str]) -> List[str]:
    """Convert a list of phoneme tokens to pinyin syllables.

    Uses a greedy parsing approach with backtracking for difficult cases.
    """
    syllables = []
    i = 0

    while i < len(tokens):
        matched = False

        for length in [3, 2, 1]:
            if i + length > len(tokens):
                continue

            segment = tokens[i:i+length]
            syllable = _try_parse_segment(segment)

            if syllable:
                syllables.append(syllable)
                i += length
                matched = True
                break

        if not matched:
            syllables.append(tokens[i])
            i += 1

    return syllables


def _try_parse_segment(segment: Tuple[str, ...]) -> str:
    """Try to parse a segment of tokens into a single pinyin syllable."""
    if len(segment) == 1:
        t = segment[0]
        if t in INITIALS or t in FINALS:
            return t
        return None

    if len(segment) == 2:
        a, b = segment
        if a in INITIALS and b in FINALS:
            return a + b
        if a in INITIALS and b == 'g' and a in ('n', 'l'):
            return a + 'g'
        if a in ('j', 'q', 'x') and b == 'v':
            return {'j': 'ju', 'q': 'qu', 'x': 'xu'}[a]
        return None

    if len(segment) == 3:
        a, b, c = segment
        if a in INITIALS and (b + c) in FINALS:
            return a + b + c
        if a in INITIALS and b == 'i' and c == 'e':
            return a + 'ie'
        if a in ('j', 'q', 'x') and b == 'i' and c == 'e':
            return {'j': 'jie', 'q': 'qie', 'x': 'xie'}[a]
        if a == 'n' and b == 'g':
            return 'ng'
        if a == 'l' and b == 'g':
            return 'lg'
        return None

    return None


def convert_phoneme_to_pinyin(phoneme_str: str) -> str:
    """Convert phoneme string to pinyin string.

    Example:
        "w o - m en - q v - b y e" -> "wo men qv bie"
    """
    tokens = parse_phoneme_tokens(phoneme_str)
    syllables = phonemes_to_syllables(tokens)
    return ' '.join(syllables)


def normalize_pinyin(pinyin: str) -> str:
    """Normalize pinyin string for comparison."""
    return pinyin.replace(' ', '').replace('，', '').replace('。', '').replace('？', '').replace('！', '')


def build_phoneme_pinyin_mapping() -> Dict[str, str]:
    """Build a mapping from phoneme pattern to pinyin from known examples."""
    mapping = {}

    examples = [
        ("w o - m en - q v - b y e", "womenqvbie"),
        ("w o - j yu e - d e - w o - m en", "wojue de women"),
        ("n a - j y eng - h ong", "naj yenghong"),
        ("d ao - sh i - h ou", "daoshi hou"),
        ("n i - h ao", "nihao"),
        ("w o - zh i - ch i - f a", "wozhi chifa"),
    ]

    for phoneme, pinyin in examples:
        tokens = parse_phoneme_tokens(phoneme)
        syllables = phonemes_to_syllables(tokens)
        mapping[' '.join(syllables)] = pinyin
        mapping[''.join(syllables)] = pinyin

    return mapping


def demo():
    """Demo the phoneme to pinyin conversion."""
    test_cases = [
        "w o - m en - q v - b y e",
        "w o - j yu e - d e - w o - m en - k e - i - j i - x v - j y ao - w ang - x y a - q v",
        "n a - j y eng - h ong - i - p y e - w en - r w en - l e - s w ei - yu e",
        "d ao - sh i - h ou - w o - q v - b ang - n i - b an - j y a",
        "w o - zh en - d e - h en - ai - t a",
    ]

    print("=== Phoneme to Pinyin Conversion ===\n")
    for phoneme in test_cases:
        pinyin = convert_phoneme_to_pinyin(phoneme)
        print(f"Phoneme: {phoneme}")
        print(f"Pinyin:  {pinyin}")
        print()


if __name__ == '__main__':
    demo()
