#!/usr/bin/env python3
"""
MCCSD Phoneme → Standard Pinyin Converter (v3)

The MCCSD dataset phoneme system uses:
  - Standard Chinese initials: b p m f d t n l g k h j q x zh ch sh r z c s
  - Medials: w (u-type), y (i-type), v (ü-type), yu (ü-type for j/q/x)
  - Finals: standard vowel/nasal/lateral finals

Syllable grouping from phoneme_intervals: [initial?][medial?][final]
  e.g.  d w an → duan,  j y e → jie,  m y eng → ming,  h w o → huo
        w o   → wo,     y ou  → you,  q v    → qu,     u     → wu
"""

INITIALS = frozenset([
    'zh', 'ch', 'sh',
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'j', 'q', 'x',
    'r', 'z', 'c', 's',
])

MEDIALS = frozenset(['w', 'y', 'v', 'yu'])

FINALS = frozenset([
    'a', 'o', 'e', 'i', 'u', 'v', 'er',
    'ai', 'ei', 'ao', 'ou',
    'an', 'en', 'ang', 'eng', 'ong',
    'ia', 'ua', 'uo', 'ie', 've',
    'iao', 'iu', 'uai', 'ui',
    'ian', 'uan', 'van',
    'in', 'un', 'vn',
    'iang', 'uang', 'ing', 'iong',
    'ueng', 'uen', 'uei',
])

# ── After-initial medial maps (transform medial+final into combined final) ────
# Used only in 3-token syllables: [initial, medial, final]
AFTER_INITIAL_W = {   # w (u-medial) after an initial
    'o':   'uo',   # h+w+o → huo
    'a':   'ua',   # zh+w+a → zhua (rare)
    'ai':  'uai',
    'ei':  'ui',   # z+w+ei → zui
    'an':  'uan',  # d+w+an → duan
    'en':  'un',   # ch+w+en → chun
    'ang': 'uang',
    'eng': 'ong',  # w+eng → ong in some contexts; standalone weng stays weng
    'i':   'ui',
    '':    'u',
}

AFTER_INITIAL_Y = {   # y (i-medial) after an initial
    'e':   'ie',   # j+y+e → jie
    'a':   'ia',   # x+y+a → xia
    'ao':  'iao',
    'ou':  'iu',   # j+y+ou → jiu
    'an':  'ian',  # j+y+an → jian
    'en':  'in',   # m+y+en → min (ien→in)
    'ang': 'iang',
    'eng': 'ing',  # m+y+eng → ming
    'ong': 'iong',
    'u':   'iu',
    '':    'i',
}

AFTER_INITIAL_V = {   # v (ü-medial) after an initial
    '':    'ü',
    'n':   'ün',
    'an':  'üan',  # → uan after j/q/x
    'e':   'üe',   # → ue after j/q/x
}

AFTER_INITIAL_YU = {  # yu after j/q/x (represents ü sound)
    '':    'u',    # j+yu → ju
    'en':  'uan',  # j+yu+en → juan
    'an':  'uan',  # q+yu+an → quan
    'e':   'ue',   # x+yu+e → xue (rare)
    'n':   'un',
}

# ── Standalone medial maps (medial at start, no initial) ─────────────────────
# Used in 2-token syllables: [medial, final]
STANDALONE_W = {
    'o':   'wo',   # w+o = wo  (NOT wuo)
    'a':   'wa',
    'ai':  'wai',
    'ei':  'wei',
    'an':  'wan',
    'en':  'wen',
    'ang': 'wang',
    'eng': 'weng',
    'u':   'wu',
    'i':   'wei',  # rare
}

STANDALONE_Y = {
    'e':   'ye',
    'a':   'ya',
    'ao':  'yao',
    'ou':  'you',  # y+ou = you (NOT yiu)
    'an':  'yan',
    'en':  'yin',  # y+en = yin
    'ang': 'yang',
    'eng': 'ying', # y+eng = ying
    'ong': 'yong',
    'i':   'yi',
    'u':   'yu',
    'ia':  'ya',
    'ie':  'ye',
    'ao':  'yao',
    'iao': 'yao',
}


def _tokenize(phoneme_str: str):
    return [t.strip() for t in phoneme_str.replace('-', ' ').split() if t.strip()]


def _parse_syllables(tokens):
    """Greedy left-to-right syllable grouper."""
    syllables = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        if tok in INITIALS:
            nxt1 = tokens[i + 1] if i + 1 < n else None
            nxt2 = tokens[i + 2] if i + 2 < n else None

            if nxt1 in MEDIALS:
                # Decide if nxt2 is a final of this syllable or start of next
                if nxt2 is not None and nxt2 not in INITIALS and nxt2 not in MEDIALS:
                    syllables.append([tok, nxt1, nxt2])
                    i += 3
                else:
                    # initial + medial only (e.g. q v, j yu)
                    syllables.append([tok, nxt1])
                    i += 2
            elif nxt1 is not None and nxt1 not in INITIALS and nxt1 not in MEDIALS:
                syllables.append([tok, nxt1])
                i += 2
            else:
                syllables.append([tok])
                i += 1

        elif tok in MEDIALS:
            nxt1 = tokens[i + 1] if i + 1 < n else None
            if nxt1 is not None and nxt1 not in INITIALS and nxt1 not in MEDIALS:
                syllables.append([tok, nxt1])
                i += 2
            else:
                syllables.append([tok])
                i += 1

        else:
            # final or standalone vowel
            syllables.append([tok])
            i += 1

    return syllables


def _syllable_to_pinyin(group):
    """Convert one syllable group to standard pinyin."""

    # ── 1 token ──────────────────────────────────────────────────────────────
    if len(group) == 1:
        tok = group[0]
        standalone1 = {
            'u': 'wu', 'i': 'yi', 'v': 'yu',
            'w': 'wu', 'y': 'yi', 'yu': 'yu',
        }
        return standalone1.get(tok, tok)

    # ── 2 tokens ─────────────────────────────────────────────────────────────
    if len(group) == 2:
        t0, t1 = group

        # initial + final
        if t0 in INITIALS and t1 not in MEDIALS:
            return t0 + t1

        # initial + medial (no explicit final)
        if t0 in INITIALS and t1 in MEDIALS:
            if t1 == 'v':
                # j/q/x: ü→u; l/n: ü stays ü (written lü/nü)
                return t0 + ('u' if t0 in ('j', 'q', 'x') else 'ü')
            if t1 == 'yu':
                return t0 + 'u'   # j+yu=ju, q+yu=qu, x+yu=xu
            if t1 == 'w':
                return t0 + 'u'
            if t1 == 'y':
                return t0 + 'i'

        # standalone medial + final
        if t0 == 'w':
            return STANDALONE_W.get(t1, 'w' + t1)
        if t0 == 'y':
            return STANDALONE_Y.get(t1, 'y' + t1)
        if t0 == 'v':
            cf = AFTER_INITIAL_V.get(t1, 'ü' + t1)
            return 'yu' + (cf[1:] if cf.startswith('ü') else cf)
        if t0 == 'yu':
            return 'yu' + t1

        return ''.join(group)

    # ── 3 tokens ─────────────────────────────────────────────────────────────
    if len(group) == 3:
        t0, t1, t2 = group
        # t0 = initial, t1 = medial, t2 = final
        if t1 == 'w':
            cf = AFTER_INITIAL_W.get(t2, 'u' + t2)
        elif t1 == 'y':
            cf = AFTER_INITIAL_Y.get(t2, 'i' + t2)
        elif t1 == 'v':
            cf = AFTER_INITIAL_V.get(t2, 'ü' + t2)
            if t0 in ('j', 'q', 'x', 'y'):
                cf = cf.replace('ü', 'u')
        elif t1 == 'yu':
            cf = AFTER_INITIAL_YU.get(t2, t2)
        else:
            cf = t1 + t2
        return t0 + cf

    return ''.join(group)


def phoneme_to_syllables(phoneme_str: str):
    tokens = _tokenize(phoneme_str)
    groups = _parse_syllables(tokens)
    return [_syllable_to_pinyin(g) for g in groups]


def phoneme_to_pinyin_spaced(phoneme_str: str) -> str:
    return ' '.join(phoneme_to_syllables(phoneme_str))


def phoneme_to_pinyin_raw(phoneme_str: str) -> str:
    return ''.join(phoneme_to_syllables(phoneme_str))


# kept for backward compatibility
def convert_phoneme_to_pinyin(phoneme_str: str) -> str:
    return phoneme_to_pinyin_spaced(phoneme_str)


def convert_phoneme_to_pinyin_raw(phoneme_str: str) -> str:
    return phoneme_to_pinyin_raw(phoneme_str)


if __name__ == '__main__':
    cases = [
        ('d w an u j y e z w ei zh u m y eng d e h w o d ong sh i s ai l ong zh ou',
         'duan wu jie zui zhu ming de huo dong shi sai long zhou'),
        ('w o m ei y ou an zh w ang zh e g e r w an j y an',
         'wo mei you an zhuang zhe ge ruan jian'),
        ('h w en l i y ao m y eng n y an ch w en t y an j v b an',
         'hun li yao ming nian chun tian ju ban'),
        ('d ao sh i h ou w o q v b ang n i b an j y a',
         'dao shi hou wo qu bang ni ban jia'),
        ('w o zh i ch i f a y an y eng g ai sh i x i j yu en g an r an',
         'wo zhi chi fa yan ying gai shi xi juan gan ran'),
        ('w o d w ei g w o h w a h en y ou x y eng q v',
         'wo dui guo hua hen you xing qu'),
    ]
    print('=== phoneme_to_pinyin_v3 self-test ===')
    all_pass = True
    for phoneme, expected in cases:
        result = phoneme_to_pinyin_spaced(phoneme)
        ok = '✓' if result == expected else '✗'
        if result != expected:
            all_pass = False
        print(f'{ok}  {phoneme}')
        if result != expected:
            print(f'    got:      {result}')
            print(f'    expected: {expected}')
    print('\nAll pass:', all_pass)
