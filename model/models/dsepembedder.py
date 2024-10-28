import torch
from torch import nn
import torch.nn.functional as F

from ._dsep_block import _DSepBlock


class _PosEmp(torch.nn.Module):
    def __init__(self, ch=7, length=60):
        super(_PosEmp, self).__init__()
        self.emb = nn.Embedding(length, ch)
        self.length = length
        self.ch = ch
        
    def forward(self, x, training=True):
        return x + self.emb(torch.arange(self.length, device=next(self.parameters()).device)).permute(1,0).unsqueeze(0)



class DSepEmbedder(torch.nn.Module):
    def __init__(self, ch=7, feature=16, length=60, padding=1):
        super(DSepEmbedder, self).__init__()
        self.despBlocks = nn.ModuleList()
        self.emb = _PosEmp(ch=ch, length=length)
        in_ch = ch
        out_ch = feature
        for i in range(6):
            self.despBlocks.append(_DSepBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch = out_ch*2
        
    def forward(self, x, training=True, l2Norm = True):
        x = x.permute(0, 2, 1)
        x = self.emb(x)
        
        for layer in self.despBlocks:
            x = layer(x)
            
        x = x.mean(axis=2)
        
        if l2Norm:
            x = x / (torch.sum(x**2,1)**(1/2)).unsqueeze(1)
            
        return x 