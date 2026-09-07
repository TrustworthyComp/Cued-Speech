import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import Counter


SPECIAL_TOKENS = ['<blank>', '<sos>', '<eos>', '<pad>']


def _tokenize_phoneme_str(phoneme_str: str) -> List[str]:
    return [t.strip() for t in phoneme_str.replace('-', ' ').split() if t.strip()]


def build_phoneme_vocab_from_file(label_paths: List[str]) -> Dict[str, int]:
    counter = Counter()
    for path in label_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                gloss = ','.join(parts[2:]).strip()
                tokens = _tokenize_phoneme_str(gloss)
                counter.update(tokens)

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


DEFAULT_PHONEME_VOCAB = build_phoneme_vocab_from_file([
    '/home/uic/fengling/mccsd/multi_speaker_train_labels_6H.txt',
    '/home/uic/fengling/mccsd/multi_speaker_test_labels_6H.txt',
])


class PhonemeTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab if vocab is not None else DEFAULT_PHONEME_VOCAB
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.blank_id = self.vocab['<blank>']
        self.sos_id = self.vocab['<sos>']
        self.eos_id = self.vocab['<eos>']
        self.pad_id = self.vocab['<pad>']
        self.vocab_size = len(self.vocab)

    def encode(self, phoneme_str: str, add_sos_eos: bool = True) -> List[int]:
        tokens = _tokenize_phoneme_str(phoneme_str)
        ids = [self.vocab.get(t, self.vocab['<blank>']) for t in tokens]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, '<unk>')
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return ' '.join(tokens)

    def collate(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = [torch.tensor(self.encode(s), dtype=torch.long) for s in sequences]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=self.pad_id)
        mask = padded != self.pad_id
        return padded, lengths, mask

    def collate_ctc(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = [torch.tensor(self.encode(s, add_sos_eos=False), dtype=torch.long) for s in sequences]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=self.pad_id)
        return padded, lengths
