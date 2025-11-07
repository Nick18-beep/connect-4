# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from connect4 import ROWS, COLS

# --- NUOVA CLASSE: Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- BLOCCO RESIDUO MODIFICATO ---
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels)  # Aggiunto blocco SE

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)  # Applicato blocco SE
        return F.relu(x + y)

class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 128, n_blocks: int = 12): #128 12  96 8
        super().__init__()
        in_planes = 4  # cur, opp, to_play, last_move
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * ROWS * COLS, COLS)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn   = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(ROWS * COLS, channels)
        self.v_fc2  = nn.Linear(channels, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        # policy
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)
        # value
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(-1)