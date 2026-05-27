from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from cms import CMSBlock


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, dim: int = 128, depth: int = 2, nhead: int = 4, max_len: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True, dim_feedforward=dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, : x.size(1)]
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class TransformerCMSClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, dim: int = 128, depth: int = 2, nhead: int = 4, max_len: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, nhead, batch_first=True),
                        "norm1": nn.LayerNorm(dim),
                        "cms": CMSBlock(dim=dim, hidden_dim=dim * 4, slow_update_every=4),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
            )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, : x.size(1)]
        for block in self.blocks:
            attn_out, _ = block["attn"](x, x, x, need_weights=False)
            x = block["norm1"](x + attn_out)
            x = block["cms"](x)
            x = block["norm2"](x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)
