from __future__ import annotations

import torch
from torch import nn


class CMSBlock(nn.Module):
    """A minimal Continuum Memory System block.

    This prototype uses two time scales:
    - fast memory: updated every forward pass
    - slow memory: updated every `slow_update_every` steps
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, slow_update_every: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.slow_update_every = slow_update_every
        self.fast_gate = nn.Linear(dim, dim)
        self.fast_updater = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.slow_updater = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("slow_memory", torch.zeros(1, 1, dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.step += 1
        residual = x
        x = self.norm(x)

        fast = torch.sigmoid(self.fast_gate(x)) * self.fast_updater(x)
        if self.slow_memory.shape[0] != x.shape[0] or self.slow_memory.shape[1] != x.shape[1]:
            self.slow_memory = torch.zeros_like(x)

        if int(self.step.item()) % self.slow_update_every == 0:
            self.slow_memory = 0.9 * self.slow_memory + 0.1 * self.slow_updater(x).detach()

        out = residual + self.dropout(fast + self.slow_memory)
        return out
