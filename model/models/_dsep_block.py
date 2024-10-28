import torch
from torch import nn
import torch.nn.functional as F

from .aiosa import LSTM, AIOSANODROP


class _RVarDSepBlock(torch.nn.Module):
    # Uses LSTM with var dropout
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(_RVarDSepBlock, self).__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv2 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        
        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch

        self.lstm = LSTM(input_size=out_ch, hidden_size=out_ch, batch_first=True, dropouti=0.1)  #0.1 dropout
        
    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.add(x, y)
        x = self.conv2(x)
        
        # added operations
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        return x


class _DSep2Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(_DSep2Block, self).__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv2 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        
        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch
        
    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.add(x, y)
        return x



class _RDSepBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(_RDSepBlock, self).__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv2 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        
        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch

        # should be named bilstm for mesa dsepnetst7
        self.lstm = nn.LSTM(input_size=out_ch, hidden_size=out_ch, batch_first=True) # added layer (actually unidirectional)
        
    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.add(x, y)
        x = self.conv2(x)
        
        # added operations
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x) # bilstm for mesa dsepnetst7
        x = x.permute(0, 2, 1)

        return x



class _DSepBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(_DSepBlock, self).__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv2 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        
        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch
        
    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.add(x, y)
        x = self.conv2(x)
        return x


class _DSepDOBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(_DSepDOBlock, self).__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv2 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        self.do = torch.nn.Dropout(0.2)
        
        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch
        
    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.do(x)
        x = torch.add(x, y)
        x = self.conv2(x)
        return x

    


# # exclude lstm for replacing spatial dropout   
class _RDSepSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(_RDSepSPBlock, self).__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv01 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch
        
    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.add(x, y)
        x = self.conv01(x)

        return x



