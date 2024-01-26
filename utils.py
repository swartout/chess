import itertools
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from fenparser import FenParser


class FenDataset(Dataset):
    def __init__(self, fens: List[str], tokenizer: BertTokenizer, max_len: int = 64):
        self.fens = fens
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer.encode_plus(
            self.fens[idx],
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        target = fen_to_target(self.fens[idx])
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'token_type_ids': torch.tensor(inputs['token_type_ids']),
            'targets': target
        }
        

def fen_to_target(f: str) -> torch.Tensor:
    p = FenParser(f)
    b = list(itertools.chain.from_iterable(p.parse()))
    pieces = ['p', 'r', 'n', 'b', 'q', 'k', 'P', 'R', 'N', 'B', 'Q', 'K']
    out = torch.zeros(768)
    for i in range(12):
        target = torch.zeros(64)
        target[[pieces[i] == x for x in b]] = 1
        out[64*i:64*(i+1)] = target
    return out

