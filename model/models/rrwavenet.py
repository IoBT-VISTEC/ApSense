import torch
from torch import nn
import torch.nn.functional as F

from .mini_rrwavenet import _MultiConv, _ResConv


class RRWaveNet(nn.Module):
    def __init__(self, num_channels=7, winsize=60):
        super(RRWaveNet, self).__init__()
        self.mtconv1 = _MultiConv(16, num_channels=num_channels, winsize=winsize)
        self.mtconv2 = _MultiConv(32, num_channels=num_channels, winsize=winsize)
        self.mtconv3 = _MultiConv(64, num_channels=num_channels, winsize=winsize)
        
        layers = []
        current_channels = num_channels * 3
        for channels in [64, 64, 128, 128, 256, 256, 512, 512]:
            layers.append(_ResConv(current_channels, channels))
            current_channels += channels
        
        self.resblocks = nn.Sequential(*layers)
        self.rl1 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=current_channels, out_features=128)
        self.rl2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.th = nn.Tanh()
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        xmt1 = self.mtconv1(x)
        xmt2 = self.mtconv2(x)
        xmt3 = self.mtconv3(x)
        x = torch.cat([xmt1, xmt2, xmt3], dim=1)
        x = self.resblocks(x)
        x = torch.mean(x, dim=2)
        x = self.rl1(x)
        x = self.fc1(x)
        x = self.rl2(x)
        x = self.fc2(x)
        x = self.th(x)
        x = self.fc3(x)
        x = torch.squeeze(x)
        return x

