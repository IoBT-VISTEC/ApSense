import torch
from torch import nn
import torch.nn.functional as F

class _MultiConv(nn.Module):
    def __init__(self, kernel_size, num_channels=7, winsize=60):
        super(_MultiConv, self).__init__() 
        self.conv = nn.Conv1d(
            in_channels=num_channels, 
            out_channels=num_channels, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.bn = nn.BatchNorm1d(num_features=num_channels)
        self.rl = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        x = self.mp(x)
        return x

class _ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ResConv, self).__init__() 
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        
    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.cat([y, x], dim=1)
        return x

class MiniRRWaveNet(nn.Module):
    def __init__(self, num_channels=7, winsize=60):
        super(MiniRRWaveNet, self).__init__()
        self.mtconv1 = _MultiConv(16, num_channels=num_channels, winsize=winsize)
        self.mtconv2 = _MultiConv(32, num_channels=num_channels, winsize=winsize)
        self.mtconv3 = _MultiConv(64, num_channels=num_channels, winsize=winsize)
        
        layers = []
        current_channels = num_channels * 3
        for channels in [64, 128]:
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

