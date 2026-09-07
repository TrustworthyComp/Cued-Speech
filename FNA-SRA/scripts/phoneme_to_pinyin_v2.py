#!/usr/bin/env python3
"""Robust Phoneme-to-Pinyin Converter (v2)"""

INITIALS = {'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
            'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's'}

FINALS_COMPOUND = {
    'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
    'an', 'en', 'in', 'un', 'vn',
    'ang', 'eng', 'ing', 'ong',
}


def parse_phoneme_tokens(phoneme_str):
    return [t.strip() for t in phoneme_str.replace('-', ' ').split() if t.strip()]


def phonemes_to_syllables_v2(tokens):
    syllables = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == 'ng':
            if syllables and syllables[-1] in ('a', 'o', 'e', 'i', 'u', 'ü'):
                syllables[-1] = syllables[-1] + 'ng'
            else:
                syllables.append('ng')
            i += 1
            continue

        if i + 2 < len(tokens):
            triple = token + tokens[i+1] + tokens[i+2]
            if triple in ('jve', 'qve', 'xve'):
                syllables.append({'jve': 'jue', 'qve': 'que', 'xve': 'xue'}[triple])
                i += 3
                continue
            if token in ('b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                         'z', 'c', 's', 'r') and tokens[i+1] == 'y' and tokens[i+2] == 'e':
                syllables.append(token + 'ie')
                i += 3
                continue

        if token in ('i', 'u', 'v'):
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if next_tok in ('a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng'):
                    syllables.append(token)
                    i += 1
                    continue
                elif next_tok == 'ng' and token == 'n':
                    syllables.append('ng')
                    i += 2
                    continue
                elif next_tok == 'g' and token in ('n', 'l'):
                    syllables.append(token + 'g')
                    i += 2
                    continue
                elif token == 'v' and next_tok not in ('a', 'o', 'e', 'i', 'u'):
                    syllables.append('ü')
                    i += 1
                    continue
            if token == 'v':
                syllables.append('ü')
            else:
                syllables.append(token)
            i += 1
            continue

        if i + 1 >= len(tokens):
            syllables.append(token.replace('v', 'ü'))
            i += 1
            continue

        next_tok = tokens[i + 1]
        compound = token + next_tok
        if compound in FINALS_COMPOUND:
            syllables.append(compound.replace('v', 'ü'))
            i += 2
            continue

        if token == 'v':
            syllables.append('ü')
            i += 1
            continue

        if token in INITIALS:
            if next_tok in ('a', 'o', 'e', 'i', 'u', 'v', 'er',
                          'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
                          'an', 'en', 'in', 'un', 'vn',
                          'ang', 'eng', 'ing', 'ong'):
                if token in ('j', 'q', 'x') and next_tok == 'v':
                    syllables.append({'j': 'ju', 'q': 'qu', 'x': 'xu'}[token])
                else:
                    syllables.append((token + next_tok).replace('v', 'ü'))
                i += 2
            elif next_tok == 'y' and i + 2 < len(tokens) and tokens[i+2] == 'e':
                syllables.append(token + 'ie')
                i += 3
            else:
                syllables.append(token.replace('v', 'ü'))
                i += 1
        else:
            syllables.append(token.replace('v', 'ü'))
            i += 1
    return syllables


def convert_phoneme_to_pinyin(phoneme_str):
    tokens = parse_phoneme_tokens(phoneme_str)
    syllables = phonemes_to_syllables_v2(tokens)
    return ' '.join(syllables)


def convert_phoneme_to_pinyin_raw(phoneme_str):
    tokens = parse_phoneme_tokens(phoneme_str)
    syllables = phonemes_to_syllables_v2(tokens)
    return ''.join(syllables)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--phoneme':
        phoneme = ' '.join(sys.argv[2:])
        print(convert_phoneme_to_pinyin(phoneme))
    else:
        test_cases = [
            ("w o - m en - q v - b y e", "wo men qv bie"),
            ("d ao - sh i - h ou", "dao shi hou"),
        ]
        for phoneme, expected in test_cases:
            pinyin = convert_phoneme_to_pinyin(phoneme)
            print(f"Input: {phoneme}")
            print(f"Output: {pinyin}")
            print()
