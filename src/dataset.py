from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .representation import ChoraleTokenizer


def load_chorale_csv(path: Path) -> List[List[int]]:
    df = pd.read_csv(path)
    notes = df[["note0", "note1", "note2", "note3"]].values.tolist()
    return [list(map(int, row)) for row in notes]


def load_chorale_folder(data_dir: Path) -> List[List[List[int]]]:
    chorales: List[List[List[int]]] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        chorales.append(load_chorale_csv(csv_path))
    return chorales


@dataclass
class CausalDataset(Dataset):
    sequences: List[List[int]]
    seq_len: int
    stride: int
    pad_id: int

    def __post_init__(self) -> None:
        self.index: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            max_start = max(0, len(seq) - 1)
            for start in range(0, max_start, self.stride):
                end = start + self.seq_len + 1
                if end <= len(seq):
                    self.index.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        chunk = seq[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_token_sequences(
    chorales: Sequence[Sequence[Sequence[int]]],
    tokenizer: ChoraleTokenizer,
) -> List[List[int]]:
    return [tokenizer.encode_timesteps(chorale) for chorale in chorales]
