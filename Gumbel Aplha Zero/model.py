# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from connect4 import ROWS, COLS

# --- NUOVA CLASSE: Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseResidual(nn.Module):
    """Depthwise-separable residual block with SE attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        y = F.relu(self.dw_bn(self.dw(x)))
        y = self.pw_bn(self.pw(y))
        y = self.se(y)
        return F.relu(x + y)

class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 128, n_blocks: int = 12):
        super().__init__()
        in_planes = 4  # cur, opp, to_play, last_move
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[DepthwiseResidual(channels) for _ in range(n_blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * ROWS * COLS, COLS)
        self.col_embed = nn.Sequential(
            nn.Linear(COLS, COLS, bias=False),
            nn.ReLU(inplace=True),
        )

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_ln   = nn.LayerNorm(ROWS * COLS)
        self.v_fc1  = nn.Linear(ROWS * COLS, channels)
        self.v_fc2  = nn.Linear(channels, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)

        occ = x[:, 0] + x[:, 1]
        col_heights = occ.sum(dim=1)
        col_bias = self.col_embed(col_heights)

        # policy
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p) + col_bias

        # value
        v = self.v_conv(h).view(h.size(0), -1)
        v = self.v_ln(v)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(-1)
