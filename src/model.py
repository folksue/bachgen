from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.2
    max_len: int = 2048


class ChoraleTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embed(input_ids) + self.pos_embed(positions)
        mask = self._causal_mask(seq_len, input_ids.device)
        hidden = self.transformer(hidden, mask=mask)
        return self.lm_head(hidden)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
