"""Multi-head Q-network for bootstrapped DDQN over legal candidates.

A shared torso (default 22 -> 128 -> 128, ReLU) feeds K linear heads. Input is a stack of per-candidate
feature vectors [N, D]; output is [N, K]: the Q-value of every candidate under every head.
Same feature layout as red.rl.ddqn.network (FEATURE_DIM = 22). No CybORG imports.
"""
import torch
from torch import nn

FEATURE_DIM = 22


class MultiHeadQNet(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden=(128, 128), heads=5):
        super().__init__()
        if heads < 1:
            raise ValueError('heads must be >= 1')
        layers, width = [], feature_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.torso = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(heads)])

    @property
    def num_heads(self):
        return len(self.heads)

    def forward(self, x):
        z = self.torso(x)
        return torch.cat([head(z) for head in self.heads], dim=-1)
