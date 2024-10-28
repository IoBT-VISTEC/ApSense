import torch
from torch import nn
import torch.nn.functional as F

from ._dsep_block import _DSepBlock, _DSepDOBlock, _RDSepBlock, _DSep2Block, _RVarDSepBlock, _RDSepSPBlock
from .aiosa import LSTM



class DSepST15Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x



class DSepST14Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST14Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 16)
        self.dsep1 = _DSepBlock(16, 16)
        self.dsep2 = _DSepBlock(16, 16)
        self.dsep3 = _DSepBlock(16, 16)

        dsep_out_ch = 16
        self.lstm = LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=1,
                         bidirectional=True, batch_first=True, dropouti=0.1)
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(2 * hidden_size, 60)
        self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(60, 1)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = torch.add(x, self.dsep1(x))
        x = torch.add(x, self.dsep2(x))
        x = torch.add(x, self.dsep3(x))
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        x = torch.flatten(x, start_dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST13Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST13Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 16)
        self.dsep1 = _DSepBlock(16, 16)
        self.dsep2 = _DSepBlock(16, 16)
        self.dsep3 = _DSepBlock(16, 16)

        dsep_out_ch = 16
        self.lstm = LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=1,
                         bidirectional=True, batch_first=True, dropouti=0.1)
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(2 * hidden_size, 60)
        self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(60, 1)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        x = torch.flatten(x, start_dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST12Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST12Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)

        dsep_out_ch = 64
        self.lstm = LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=1,
                         bidirectional=True, batch_first=True, dropouti=0.1)
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(2 * hidden_size, 60)
        self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(60, 1)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        x = torch.flatten(x, start_dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST11Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST11Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)

        dsep_out_ch = 64
        self.lstm = LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=1,
                         bidirectional=True, batch_first=True, dropouti=0.1)
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + hidden_size * 2 * hidden_size2, 60)
        self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(60, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
#         x = x[:, -1, :]
        
        # Lower branch (permutation)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], dim=1)
        
        x = self.dense1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST10Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST10Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSep2Block(64, 64)
        self.dsep2 = _DSep2Block(64, 64)
        self.dsep3 = _DSep2Block(64, 64)

        dsep_out_ch = 64
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 256)
        self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 60)
        self.dense3 = nn.Linear(60, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], dim=1)
        
        x = self.dense1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST9Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST9Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 128)
        self.dsep2 = _DSepBlock(128, 64)

        dsep_out_ch = 64
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 256)
#         self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 60)
        self.dense3 = nn.Linear(60, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
#         x = self.dsep3(x)
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], dim=1)
        
        x = self.dense1(x)
        x = self.relu(x)
#         x = self.do(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST8Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST8Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)

        dsep_out_ch = 64
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 256)
        self.do = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 60)
        self.dense3 = nn.Linear(60, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)

#         self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], dim=1)
        
        x = self.dense1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST7Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
        self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
        self.dsep12 = _RDSepBlock(32, 32)
        self.dsep13 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
        self.dsep22 = _RDSepBlock(16, 16)
        self.dsep23 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
        self.dsep32 = _RDSepBlock(8, 8)
        self.dsep33 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x




class DSepST6Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST6Net, self).__init__()

        in_ch = num_channels

        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)
            
        dsep_out_ch = 64

        self.rnn = nn.LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=layer, dropout=dropout, batch_first=True, bidirectional=True)
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2 + hidden_size * 2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        # Middle branch
        z = x.permute(0, 2, 1)
        self.rnn.flatten_parameters()
        z, _ = self.rnn(z)
        z = torch.mean(z, 1)
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        z = torch.flatten(z, start_dim=1)
        x = torch.cat([x, y, z], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x



class DSepST5Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST5Net, self).__init__()

        in_ch = num_channels

        self.small_conv = nn.Conv1d(in_ch, in_ch, 3, padding="same")
        self.large_conv = nn.Conv1d(in_ch, in_ch, 30, padding="same")
        
        self.dsep0 = _DSepBlock(in_ch*2, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)
            
        dsep_out_ch = 64

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        a = self.small_conv(x)
        b = self.large_conv(x)
        x = torch.cat([a, b], 1)
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST4Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST4Net, self).__init__()

        self.dsep = []
        
        self.indices = torch.LongTensor(range(winsize))
        in_ch = num_channels
            
        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)
            
        dsep_out_ch = 64
        
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, batch_first=True)

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        self.softm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x



class DSepST3Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST3Net, self).__init__()

        self.dsep = []

        in_ch = num_channels
            
        self.dsep0 = _DSepDOBlock(in_ch, 64)
        self.dsep1 = _DSepDOBlock(64, 64)
        self.dsep2 = _DSepDOBlock(64, 64)
        self.dsep3 = _DSepDOBlock(64, 64)
            
        dsep_out_ch = 64

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST2Net(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST2Net, self).__init__()

        self.dsep = []
        
        self.indices = torch.LongTensor(range(winsize))
        in_ch = num_channels
            
        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 64)
        self.dsep2 = _DSepBlock(64, 64)
        self.dsep3 = _DSepBlock(64, 64)
            
        dsep_out_ch = 64

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        self.softm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepSTNet(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepSTNet, self).__init__()

        self.dsep = []
        
        self.indices = torch.LongTensor(range(winsize))
        in_ch = num_channels
            
        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 128)
        self.dsep2 = _DSepBlock(128, 256)
        self.dsep3 = _DSepBlock(256, 64)
            
        dsep_out_ch = 64

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        self.softm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x



class DSepNet(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepNet, self).__init__()

        conv_channels = 5
        self.dsep = []
        
        self.indices = torch.LongTensor(range(winsize))
        in_ch = num_channels
            
        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 128)
        self.dsep2 = _DSepBlock(128, 256)
        self.dsep3 = _DSepBlock(256, 64)
            
        dsep_out_ch = 64

        self.rnn = nn.LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=layer, dropout=dropout, batch_first=True, bidirectional=True)

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size * 2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        self.softm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)
        x = self.dsep3(x)

        x = x.permute(0, 2, 1)
        
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)
        
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)

        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

class DSepNetSmall(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepNetSmall, self).__init__()

        conv_channels = 5
        self.dsep = []
        
        self.indices = torch.LongTensor(range(winsize))
        in_ch = num_channels
            
        self.dsep0 = _DSepBlock(in_ch, 64)
        self.dsep1 = _DSepBlock(64, 128)
        self.dsep2 = _DSepBlock(128, 64)
            
        dsep_out_ch = 64

        self.rnn = nn.LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=layer, dropout=dropout, batch_first=True, bidirectional=True)

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size * 2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        self.softm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)
        x = self.dsep1(x)
        x = self.dsep2(x)

        x = x.permute(0, 2, 1)
        
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)
        
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)

        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepNetTiny(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepNetTiny, self).__init__()

        conv_channels = 5
        self.dsep = []
        
        self.indices = torch.LongTensor(range(winsize))
        in_ch = num_channels
            
        self.dsep0 = _DSepBlock(in_ch, 64)
            
        dsep_out_ch = 64

        self.rnn = nn.LSTM(input_size=dsep_out_ch, hidden_size=hidden_size, num_layers=layer, dropout=dropout, batch_first=True, bidirectional=True)

        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(hidden_size * 2, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size * 2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        self.softm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
                
        x = self.dsep0(x)

        x = x.permute(0, 2, 1)
        
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)
        
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)

        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


# # DsepT15Net + Skip connection created by Wan 30-10-2566
class DSepST15Net_skip(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_skip, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
    
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        # Adding a skip connection to the final DSep block
        skip_connection = x
        
        x = self.dsep30(x)
        x = self.dsep31(x)
    
        # Adding the skip connection here. This code adds a skip connection between the output of self dsep31 and the output of the 
        #previous dSep block, which is the final self dsep33
        
        x = torch.add(skip_connection, x)
        
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # code remains the same
        # Lower branch (permutation)

        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

# #created by Wan 1-11-2566        
# class DSepST15Net_skip_all_blocks2(nn.Module):
#     def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
#         super(DSepST15Net_skip_all_blocks2, self).__init__()

#         in_ch = num_channels

#         self.dsep00 = _RVarDSepBlock(in_ch, 64)
#         self.dsep01 = _RVarDSepBlock(64, 64)
#         self.dsep02 = _RVarDSepBlock(64, 64)
#         self.dsep03 = _RVarDSepBlock(64, 64)
#         self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RVarDSepBlock(32, 32)
#         self.dsep11 = _RVarDSepBlock(32, 32)
#         self.dsep12 = _RVarDSepBlock(32, 32)
#         self.dsep13 = _RVarDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RVarDSepBlock(16, 16)
#         self.dsep21 = _RVarDSepBlock(16, 16)
#         self.dsep22 = _RVarDSepBlock(16, 16)
#         self.dsep23 = _RVarDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RVarDSepBlock(8, 8)
#         self.dsep31 = _RVarDSepBlock(8, 8)
#         self.dsep32 = _RVarDSepBlock(8, 8)
#         self.dsep33 = _RVarDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

#         dsep_out_ch = 4

#         self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
#         self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)

#         self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
#         self.dense2 = nn.Linear(8, 1)

#         self.tanh = torch.nn.Tanh()
#         self.relu = torch.nn.ReLU()
#         self.sigm = torch.nn.Sigmoid()
#         self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
#         self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)

#         self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)

#         skip_connections = []

#         x = self.dsep00(x)
#         skip_connections.append(x)
#         x = self.dsep01(x)
#         skip_connections.append(x)
#         x = self.dsep02(x)
#         skip_connections.append(x)
#         x = self.dsep03(x)
#         skip_connections.append(x)

#         x = x.permute(0, 2, 1)
#         x = self.avg0(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         skip_connections.append(x)
#         x = self.dsep11(x)
#         skip_connections.append(x)
#         x = self.dsep12(x)
#         skip_connections.append(x)
#         x = self.dsep13(x)
#         skip_connections.append(x)

#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         skip_connections.append(x)
#         x = self.dsep21(x)
#         skip_connections.append(x)
#         x = self.dsep22(x)
#         skip_connections.append(x)
#         x = self.dsep23(x)
#         skip_connections.append(x)

#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         skip_connections.append(x)
#         x = self.dsep31(x)
#         skip_connections.append(x)
#         x = self.dsep32(x)
#         skip_connections.append(x)
#         x = self.dsep33(x)
#         skip_connections.append(x)

#         # Implement distinct-source skip connections here
#         x1 = skip_connections[0]
#         x2 = skip_connections[4]
#         x3 = skip_connections[8]
#         x4 = skip_connections[12]

#         # Combine the skip connections
#         x = x + x1 + x2 + x3 + x4

#         # Lower branch (permutation)
#         x = x.permute(0, 2, 1)
#         y = self.convbranch_1(x)
#         y = self.bn3(y)
#         y = self.relu(y)

#         # Upper branch (no permutation)
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = torch.flatten(x, start_dim=1)
#         y = torch.flatten(y, start_dim=1)
#         x = torch.cat([x, y], 1)

#         x = self.dense1(x)
#         x = self.bn_dense_1(x)
#         x = self.relu(x)
#         x = self.dense2(x)

#         x = torch.squeeze(x)

#         return x

    


# class DSepST15Net_skip_all_blocks3(nn.Module):
#     def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
#         super(DSepST15Net_skip_all_blocks3, self).__init__()

#         in_ch = num_channels

#         self.dsep00 = _RVarDSepBlock(in_ch, 64)
#         self.dsep01 = _RVarDSepBlock(64, 64)
#         self.dsep02 = _RVarDSepBlock(64, 64)
#         self.dsep03 = _RVarDSepBlock(64, 64)
#         self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RVarDSepBlock(32, 32)
#         self.dsep11 = _RVarDSepBlock(32, 32)
#         self.dsep12 = _RVarDSepBlock(32, 32)
#         self.dsep13 = _RVarDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RVarDSepBlock(16, 16)
#         self.dsep21 = _RVarDSepBlock(16, 16)
#         self.dsep22 = _RVarDSepBlock(16, 16)
#         self.dsep23 = _RVarDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RVarDSepBlock(8, 8)
#         self.dsep31 = _RVarDSepBlock(8, 8)
#         self.dsep32 = _RVarDSepBlock(8, 8)
#         self.dsep33 = _RVarDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

#         dsep_out_ch = 4

#         self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
#         self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)

#         self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
#         self.dense2 = nn.Linear(8, 1)

#         self.tanh = torch.nn.Tanh()
#         self.relu = torch.nn.ReLU()
#         self.sigm = torch.nn.Sigmoid()
#         self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
#         self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)

#         self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)

#     def forward(self, x):

#         x = x.permute(0, 2, 1)

#         x = self.dsep00(x)
#         x = self.dsep01(x)
#         x_clone = x.clone()
#         x = self.dsep02(x)
#         x = self.dsep03(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg0(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         x = self.dsep11(x)
#         x_clone = x.clone()
#         x = self.dsep12(x)
#         x = self.dsep13(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x = self.dsep21(x)
#         x_clone = x.clone()
#         x = self.dsep22(x)
#         x = self.dsep23(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x = self.dsep31(x)
#         x_clone = x.clone()
#         x = self.dsep32(x)
#         x = self.dsep33(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

#         # Lower branch (permutation)
#         x = x.permute(0, 2, 1)
#         y = self.convbranch_1(x)
#         y = self.bn3(y)
#         y = self.relu(y)

#         # Upper branch (no permutation)
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = torch.flatten(x, start_dim=1)
#         y = torch.flatten(y, start_dim=1)
#         x = torch.cat([x, y], 1)

#         x = self.dense1(x)
#         x = self.bn_dense_1(x)
#         x = self.relu(x)
#         x = self.dense2(x)

#         x = torch.squeeze(x)

#         return x

    

class DSepST15Net_skip_all_blocks3(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_skip_all_blocks3, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
    
  
        x = self.dsep00(x)
        x = self.dsep01(x)
        x_clone = x.clone()
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x_clone = x.clone()
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)

        x = self.dsep20(x)
        x = self.dsep21(x)
        x_clone = x.clone()
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)

        x = self.dsep30(x)
        x = self.dsep31(x)
        x_clone = x.clone()
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)



        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x  

    

    


# class DSepST15Net_skip_all_blocks4(nn.Module):
#     def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
#         super(DSepST15Net_skip_all_blocks4, self).__init__()

#         in_ch = num_channels

#         self.dsep00 = _RVarDSepBlock(in_ch, 64)
#         self.dsep01 = _RVarDSepBlock(64, 64)
#         self.dsep02 = _RVarDSepBlock(64, 64)
#         self.dsep03 = _RVarDSepBlock(64, 64)
#         self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RVarDSepBlock(32, 32)
#         self.dsep11 = _RVarDSepBlock(32, 32)
#         self.dsep12 = _RVarDSepBlock(32, 32)
#         self.dsep13 = _RVarDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RVarDSepBlock(16, 16)
#         self.dsep21 = _RVarDSepBlock(16, 16)
#         self.dsep22 = _RVarDSepBlock(16, 16)
#         self.dsep23 = _RVarDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RVarDSepBlock(8, 8)
#         self.dsep31 = _RVarDSepBlock(8, 8)
#         self.dsep32 = _RVarDSepBlock(8, 8)
#         self.dsep33 = _RVarDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

#         dsep_out_ch = 4

#         self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
#         self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)

#         self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
#         self.dense2 = nn.Linear(8, 1)

#         self.tanh = torch.nn.Tanh()
#         self.relu = torch.nn.ReLU()
#         self.sigm = torch.nn.Sigmoid()
#         self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
#         self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)

#         self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)

#         x = self.dsep00(x)
#         x_clone = x.clone()
#         x = self.dsep01(x)
#         x = x + x_clone
#         x = self.dsep02(x)
#         x_clone = x.clone()
#         x = self.dsep03(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg0(x)
#         x = x.permute(0, 2, 1)

        

#         x = self.dsep10(x)
#         x_clone = x.clone()
#         x = self.dsep11(x)
#         x = x + x_clone
#         x = self.dsep12(x)
#         x_clone = x.clone()
#         x = self.dsep13(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x_clone = x.clone()
#         x = self.dsep21(x)
#         x = x + x_clone
#         x = self.dsep22(x)
#         x_clone = x.clone()
#         x = self.dsep23(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x_clone = x.clone()
#         x = self.dsep31(x)
#         x = x + x_clone
#         x = self.dsep32(x)
#         x_clone = x.clone()
#         x = self.dsep33(x)
#         x = x + x_clone
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

#         # Lower branch (permutation)
#         x = x.permute(0, 2, 1)
#         y = self.convbranch_1(x)
#         y = self.bn3(y)
#         y = self.relu(y)

#         # Upper branch (no permutation)
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = torch.flatten(x, start_dim=1)
#         y = torch.flatten(y, start_dim=1)
#         x = torch.cat([x, y], 1)

#         x = self.dense1(x)
#         x = self.bn_dense_1(x)
#         x = self.relu(x)
#         x = self.dense2(x)

#         x = torch.squeeze(x)

#         return x

    

# ##########################################################

class DSepST16Net_skip(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST16Net_skip, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.dsep00(x)
        skip_connections = x
        x = self.dsep01(x)
        x = torch.add(skip_connections, x)
        x = self.dsep02(x)
        x = torch.add(skip_connections, x)
        x = self.dsep03(x)
        x = torch.add(skip_connections, x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        
        x = self.dsep10(x)
        skip_connections = x
        x = self.dsep11(x)
        x = torch.add(skip_connections, x)
        x = self.dsep12(x)
        x = torch.add(skip_connections, x)
        x = self.dsep13(x)
        x = torch.add(skip_connections, x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        
        x = self.dsep20(x)
        skip_connections = x
        x = self.dsep21(x)
        x = torch.add(skip_connections, x)
        x = self.dsep22(x)
        x = torch.add(skip_connections, x)
        x = self.dsep23(x)
        x = torch.add(skip_connections, x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        
        
        x = self.dsep30(x)
        skip_connections = x
        x = self.dsep31(x)
        x = torch.add(skip_connections, x)
        x = self.dsep32(x)
        x = torch.add(skip_connections, x)
        x = self.dsep33(x)
        x = torch.add(skip_connections, x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # code remains the same
        # Lower branch (permutation)

        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    

    


class DSepST20Net_skip(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST20Net_skip, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        x = self.dsep00(x)
        skip_connections_01 = x
        x = self.dsep01(x)
        x = torch.add(skip_connections_01, x)
        skip_connections_02 = x
        x = self.dsep02(x)
        x = torch.add(skip_connections_02, x)
        skip_connections_03 = x
        x = self.dsep03(x)
        x = torch.add(skip_connections_03, x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.dsep10(x)
        skip_connections_04 = x
        x = self.dsep11(x)
        x = torch.add(skip_connections_04, x)
        skip_connections_05 = x
        x = self.dsep12(x)
        x = torch.add(skip_connections_05, x)
        skip_connections_06 = x
        x = self.dsep13(x)
        x = torch.add(skip_connections_06, x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.dsep20(x)
        skip_connections_07 = x
        x = self.dsep21(x)
        x = torch.add(skip_connections_07, x)
        skip_connections_08 = x
        x = self.dsep22(x)
        x = torch.add(skip_connections_08, x)
        skip_connections_09 = x
        x = self.dsep23(x)
        x = torch.add(skip_connections_09, x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.dsep30(x)
        skip_connections_10 = x
        x = self.dsep31(x)
        x = torch.add(skip_connections_10, x)
        skip_connections_11 = x
        x = self.dsep32(x)
        x = torch.add(skip_connections_11, x)
        skip_connections_12 = x
        x = self.dsep33(x)
        x = torch.add(skip_connections_12, x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)



        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    

    

class DSepST7Net_skip_aiosa(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_skip_aiosa, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
        self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
        self.dsep12 = _RDSepBlock(32, 32)
        self.dsep13 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
        self.dsep22 = _RDSepBlock(16, 16)
        self.dsep23 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
        self.dsep32 = _RDSepBlock(8, 8)
        self.dsep33 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        x = self.dsep00(x)
        skip_connections_01 = x
        x = self.dsep01(x)
        x = torch.add(skip_connections_01, x)
        skip_connections_02 = x
        x = self.dsep02(x)
        x = torch.add(skip_connections_02, x)
        skip_connections_03 = x
        x = self.dsep03(x)
        x = torch.add(skip_connections_03, x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.dsep10(x)
        skip_connections_04 = x
        x = self.dsep11(x)
        x = torch.add(skip_connections_04, x)
        skip_connections_05 = x
        x = self.dsep12(x)
        x = torch.add(skip_connections_05, x)
        skip_connections_06 = x
        x = self.dsep13(x)
        x = torch.add(skip_connections_06, x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.dsep20(x)
        skip_connections_07 = x
        x = self.dsep21(x)
        x = torch.add(skip_connections_07, x)
        skip_connections_08 = x
        x = self.dsep22(x)
        x = torch.add(skip_connections_08, x)
        skip_connections_09 = x
        x = self.dsep23(x)
        x = torch.add(skip_connections_09, x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.dsep30(x)
        skip_connections_10 = x
        x = self.dsep31(x)
        x = torch.add(skip_connections_10, x)
        skip_connections_11 = x
        x = self.dsep32(x)
        x = torch.add(skip_connections_11, x)
        skip_connections_12 = x
        x = self.dsep33(x)
        x = torch.add(skip_connections_12, x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)


        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    




class DSepST7Net_skip_some(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_skip_some, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
        self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
        self.dsep12 = _RDSepBlock(32, 32)
        self.dsep13 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
        self.dsep22 = _RDSepBlock(16, 16)
        self.dsep23 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
        self.dsep32 = _RDSepBlock(8, 8)
        self.dsep33 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
        
    
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
    
        x = self.dsep00(x)
        x = self.dsep01(x)
        x_clone = x.clone()
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x_clone = x.clone()
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)

        x = self.dsep20(x)
        x = self.dsep21(x)
        x_clone = x.clone()
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)

        x = self.dsep30(x)
        x = self.dsep31(x)
        x_clone = x.clone()
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x + x_clone
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)


        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    

    

    

    

# PASS     
# class DSepST7Net_spatial_dropout(nn.Module):
#     def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
#         super(DSepST7Net_spatial_dropout, self).__init__()

#         in_ch = num_channels

#         self.dsep00 = _RDSepSPBlock(in_ch, 64)
#         self.dsep01 = _RDSepSPBlock(64, 64)
#         self.dsep02 = _RDSepSPBlock(64, 64)
#         self.dsep03 = _RDSepSPBlock(64, 64)
#         self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RDSepSPBlock(32, 32)
#         self.dsep11 = _RDSepSPBlock(32, 32)
#         self.dsep12 = _RDSepSPBlock(32, 32)
#         self.dsep13 = _RDSepSPBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RDSepSPBlock(16, 16)
#         self.dsep21 = _RDSepSPBlock(16, 16)
#         self.dsep22 = _RDSepSPBlock(16, 16)
#         self.dsep23 = _RDSepSPBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RDSepSPBlock(8, 8)
#         self.dsep31 = _RDSepSPBlock(8, 8)
#         self.dsep32 = _RDSepSPBlock(8, 8)
#         self.dsep33 = _RDSepSPBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

#         dsep_out_ch = 4

#         self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
#         self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)

#         self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
#         self.dense2 = nn.Linear(8, 1)

#         self.tanh = torch.nn.Tanh()
#         self.relu = torch.nn.ReLU()
#         self.sigm = torch.nn.Sigmoid()
#         self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
#         self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)

#         self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)

#         x = self.dsep00(x)
#         x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg0(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

#         # Lower branch (permutation)
#         x = x.permute(0, 2, 1)
#         y = self.convbranch_1(x)
#         y = self.bn3(y)
#         y = self.relu(y)

#         # Upper branch (no permutation)
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = torch.flatten(x, start_dim=1)
#         y = torch.flatten(y, start_dim=1)
#         x = torch.cat([x, y], 1)

#         x = self.dense1(x)
#         x = self.bn_dense_1(x)
#         x = self.relu(x)
#         x = self.dense2(x)

#         x = torch.squeeze(x)

#         return x








class DSepST7Net_spatial_dropout(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_spatial_dropout, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepSPBlock(in_ch, 64)
        self.drop_0 = nn.Dropout2d(0.2)
        self.dsep01 = _RDSepSPBlock(64, 64)
        self.drop_1 = nn.Dropout2d(0.2)
        self.dsep02 = _RDSepSPBlock(64, 64)
        self.drop_2 = nn.Dropout2d(0.2)
        self.dsep03 = _RDSepSPBlock(64, 64)
        self.drop_3 = nn.Dropout2d(0.2)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepSPBlock(32, 32)
        self.drop_10 = nn.Dropout2d(0.1)
        self.dsep11 = _RDSepSPBlock(32, 32)
        self.drop_11 = nn.Dropout2d(0.1)
        self.dsep12 = _RDSepSPBlock(32, 32)
        self.drop_12 = nn.Dropout2d(0.1)
        self.dsep13 = _RDSepSPBlock(32, 32)
        self.drop_13 = nn.Dropout2d(0.1)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepSPBlock(16, 16)
        self.drop_20 = nn.Dropout2d(0.3)
        self.dsep21 = _RDSepSPBlock(16, 16)
        self.drop_21 = nn.Dropout2d(0.3)
        self.dsep22 = _RDSepSPBlock(16, 16)
        self.drop_22 = nn.Dropout2d(0.3)
        self.dsep23 = _RDSepSPBlock(16, 16)
        self.drop_23 = nn.Dropout2d(0.3)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepSPBlock(8, 8)
        self.drop_30 = nn.Dropout2d(0.5)
        self.dsep31 = _RDSepSPBlock(8, 8)
        self.drop_31 = nn.Dropout2d(0.5)
        self.dsep32 = _RDSepSPBlock(8, 8)
        self.drop_32 = nn.Dropout2d(0.5)
        self.dsep33 = _RDSepSPBlock(8, 8)
        self.drop_33 = nn.Dropout2d(0.5)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        
        
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.drop_0(x)
        x = self.dsep01(x)
        x = self.drop_1(x)
        x = self.dsep02(x)
        x = self.drop_2(x)
        x = self.dsep03(x)
        x = self.drop_3(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.drop_10(x)
        x = self.dsep11(x)
        x = self.drop_11(x)
        x = self.dsep12(x)
        x = self.drop_12(x)
        x = self.dsep13(x)
        x = self.drop_13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.drop_20(x)
        x = self.dsep21(x)
        x = self.drop_21(x)
        x = self.dsep22(x)
        x = self.drop_22(x)
        x = self.dsep23(x)
        x = self.drop_23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.drop_30(x)
        x = self.dsep31(x)
        x = self.drop_31(x)
        x = self.dsep32(x)
        x = self.drop_32(x)
        x = self.dsep33(x)
        x = self.drop_33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)
        

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    

    

class DSepST7Net_spatial_lstm_dropout(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_spatial_lstm_dropout, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepSPBlock(in_ch, 64)
        self.drop_0 = nn.Dropout2d(0.2)
        self.dsep01 = _RDSepSPBlock(64, 64)
        self.drop_1 = nn.Dropout2d(0.2)
        self.dsep02 = _RDSepSPBlock(64, 64)
        self.drop_2 = nn.Dropout2d(0.2)
        self.dsep03 = _RDSepSPBlock(64, 64)
        self.drop_3 = nn.Dropout2d(0.2)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepSPBlock(32, 32)
        self.drop_10 = nn.Dropout2d(0.1)
        self.dsep11 = _RDSepSPBlock(32, 32)
        self.drop_11 = nn.Dropout2d(0.1)
        self.dsep12 = _RDSepSPBlock(32, 32)
        self.drop_12 = nn.Dropout2d(0.1)
        self.dsep13 = _RDSepSPBlock(32, 32)
        self.drop_13 = nn.Dropout2d(0.1)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepSPBlock(16, 16)
        self.drop_20 = nn.Dropout2d(0.3)
        self.dsep21 = _RDSepSPBlock(16, 16)
        self.drop_21 = nn.Dropout2d(0.3)
        self.dsep22 = _RDSepSPBlock(16, 16)
        self.drop_22 = nn.Dropout2d(0.3)
        self.dsep23 = _RDSepSPBlock(16, 16)
        self.drop_23 = nn.Dropout2d(0.3)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepSPBlock(8, 8)
        self.drop_30 = nn.Dropout2d(0.5)
        self.dsep31 = _RDSepSPBlock(8, 8)
        self.drop_31 = nn.Dropout2d(0.5)
        self.dsep32 = _RDSepSPBlock(8, 8)
        self.drop_32 = nn.Dropout2d(0.5)
        self.dsep33 = _RDSepSPBlock(8, 8)
        self.drop_33 = nn.Dropout2d(0.5)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.lstm_1 = LSTM(input_size= dsep_out_ch, hidden_size= dsep_out_ch, batch_first=True, dropouti=0.1)
        
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.drop_0(x)
        x = self.dsep01(x)
        x = self.drop_1(x)
        x = self.dsep02(x)
        x = self.drop_2(x)
        x = self.dsep03(x)
        x = self.drop_3(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.drop_10(x)
        x = self.dsep11(x)
        x = self.drop_11(x)
        x = self.dsep12(x)
        x = self.drop_12(x)
        x = self.dsep13(x)
        x = self.drop_13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.drop_20(x)
        x = self.dsep21(x)
        x = self.drop_21(x)
        x = self.dsep22(x)
        x = self.drop_22(x)
        x = self.dsep23(x)
        x = self.drop_23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.drop_30(x)
        x = self.dsep31(x)
        x = self.drop_31(x)
        x = self.dsep32(x)
        x = self.drop_32(x)
        x = self.dsep33(x)
        x = self.drop_33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)
        
        
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)
        

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        #aad LSTM pass
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_1(x)
        x = x.permute(0, 2, 1)
        
   
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

class DSepST15Net_add_no_drop_lstm_final(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_add_no_drop_lstm_final, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.lstm_1 = LSTM(input_size= dsep_out_ch, hidden_size= dsep_out_ch, batch_first=True, dropouti=0.0)
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        
        #aad LSTM pass
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_1(x)
        x = x.permute(0, 2, 1)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    



    

class DSepST7Net_spatial_add_no_drop_lstm_final(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_spatial_add_no_drop_lstm_final, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepSPBlock(in_ch, 64)
        self.drop_0 = nn.Dropout2d(0.2)
        self.dsep01 = _RDSepSPBlock(64, 64)
        self.drop_1 = nn.Dropout2d(0.2)
        self.dsep02 = _RDSepSPBlock(64, 64)
        self.drop_2 = nn.Dropout2d(0.2)
        self.dsep03 = _RDSepSPBlock(64, 64)
        self.drop_3 = nn.Dropout2d(0.2)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepSPBlock(32, 32)
        self.drop_10 = nn.Dropout2d(0.1)
        self.dsep11 = _RDSepSPBlock(32, 32)
        self.drop_11 = nn.Dropout2d(0.1)
        self.dsep12 = _RDSepSPBlock(32, 32)
        self.drop_12 = nn.Dropout2d(0.1)
        self.dsep13 = _RDSepSPBlock(32, 32)
        self.drop_13 = nn.Dropout2d(0.1)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepSPBlock(16, 16)
        self.drop_20 = nn.Dropout2d(0.3)
        self.dsep21 = _RDSepSPBlock(16, 16)
        self.drop_21 = nn.Dropout2d(0.3)
        self.dsep22 = _RDSepSPBlock(16, 16)
        self.drop_22 = nn.Dropout2d(0.3)
        self.dsep23 = _RDSepSPBlock(16, 16)
        self.drop_23 = nn.Dropout2d(0.3)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepSPBlock(8, 8)
        self.drop_30 = nn.Dropout2d(0.5)
        self.dsep31 = _RDSepSPBlock(8, 8)
        self.drop_31 = nn.Dropout2d(0.5)
        self.dsep32 = _RDSepSPBlock(8, 8)
        self.drop_32 = nn.Dropout2d(0.5)
        self.dsep33 = _RDSepSPBlock(8, 8)
        self.drop_33 = nn.Dropout2d(0.5)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.lstm_1 = LSTM(input_size= dsep_out_ch, hidden_size= dsep_out_ch, batch_first=True, dropouti=0.0)
        
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.drop_0(x)
        x = self.dsep01(x)
        x = self.drop_1(x)
        x = self.dsep02(x)
        x = self.drop_2(x)
        x = self.dsep03(x)
        x = self.drop_3(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.drop_10(x)
        x = self.dsep11(x)
        x = self.drop_11(x)
        x = self.dsep12(x)
        x = self.drop_12(x)
        x = self.dsep13(x)
        x = self.drop_13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.drop_20(x)
        x = self.dsep21(x)
        x = self.drop_21(x)
        x = self.dsep22(x)
        x = self.drop_22(x)
        x = self.dsep23(x)
        x = self.drop_23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.drop_30(x)
        x = self.dsep31(x)
        x = self.drop_31(x)
        x = self.dsep32(x)
        x = self.drop_32(x)
        x = self.dsep33(x)
        x = self.drop_33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)
        
        
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)
        

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        #aad LSTM pass
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_1(x)
        x = x.permute(0, 2, 1)
        
   
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x



    

class DSepST15Net_add_drop_lstm_final(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_add_drop_lstm_final, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.lstm_1 = LSTM(input_size= dsep_out_ch, hidden_size= dsep_out_ch, batch_first=True, dropouti=0.1)
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        
        #aad LSTM pass
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_1(x)
        x = x.permute(0, 2, 1)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    

### Test spt reducing block step
class DSepST7Net_b1(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_b1, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
#         self.dsep01 = _RDSepBlock(64, 64)
#         self.dsep02 = _RDSepBlock(64, 64)
#         self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RDSepBlock(32, 32)
#         self.dsep11 = _RDSepBlock(32, 32)
#         self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RDSepBlock(16, 16)
#         self.dsep21 = _RDSepBlock(16, 16)
#         self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RDSepBlock(8, 8)
#         self.dsep31 = _RDSepBlock(8, 8)
#         self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

        dsep_out_ch = 32
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
#         x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    


class DSepST7Net_b2(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_b2, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
#         self.dsep02 = _RDSepBlock(64, 64)
#         self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RDSepBlock(32, 32)
#         self.dsep11 = _RDSepBlock(32, 32)
#         self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RDSepBlock(16, 16)
#         self.dsep21 = _RDSepBlock(16, 16)
#         self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RDSepBlock(8, 8)
#         self.dsep31 = _RDSepBlock(8, 8)
#         self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

        dsep_out_ch = 32
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x


class DSepST7Net_b3(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_b3, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
#         self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RDSepBlock(32, 32)
#         self.dsep11 = _RDSepBlock(32, 32)
#         self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RDSepBlock(16, 16)
#         self.dsep21 = _RDSepBlock(16, 16)
#         self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RDSepBlock(8, 8)
#         self.dsep31 = _RDSepBlock(8, 8)
#         self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

        dsep_out_ch = 32
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x    

    

class DSepST7Net_b5(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_b5, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
        self.dsep03 = _RDSepBlock(64, 64)
        self.dsep04 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep10 = _RDSepBlock(32, 32)
#         self.dsep11 = _RDSepBlock(32, 32)
#         self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
#         self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep20 = _RDSepBlock(16, 16)
#         self.dsep21 = _RDSepBlock(16, 16)
#         self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
#         self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

#         self.dsep30 = _RDSepBlock(8, 8)
#         self.dsep31 = _RDSepBlock(8, 8)
#         self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
#         self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

        dsep_out_ch = 32
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = self.dsep04(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

#         x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg1(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg2(x)
#         x = x.permute(0, 2, 1)

#         x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
#         x = x.permute(0, 2, 1)
#         x = self.avg3(x)
#         x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x    

    


class DSepST7Net_4b_4e_1l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_4b_4e_1l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
#         self.dsep01 = _RDSepBlock(64, 64)
#         self.dsep02 = _RDSepBlock(64, 64)
#         self.dsep03 = _RDSepBlock(64, 64)
#         self.dsep04 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
#         self.dsep11 = _RDSepBlock(32, 32)
#         self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
#         self.dsep21 = _RDSepBlock(16, 16)
#         self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
#         self.dsep31 = _RDSepBlock(8, 8)
#         self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
#         x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
#         x = self.dsep04(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x    

    

class DSepST7Net_4b_4e_2l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_4b_4e_2l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
#         self.dsep02 = _RDSepBlock(64, 64)
#         self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
#         self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
#         self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
#         self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x      

    


class DSepST7Net_4b_4e_3l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_4b_4e_3l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
#         self.dsep03 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
        self.dsep12 = _RDSepBlock(32, 32)
#         self.dsep13 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
        self.dsep22 = _RDSepBlock(16, 16)
#         self.dsep23 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
        self.dsep32 = _RDSepBlock(8, 8)
#         self.dsep33 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
#         x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
#         x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
#         x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x   

    


class DSepST7Net_4b_4e_5l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_4b_4e_5l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
        self.dsep03 = _RDSepBlock(64, 64)
        self.dsep04 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
        self.dsep12 = _RDSepBlock(32, 32)
        self.dsep13 = _RDSepBlock(32, 32)
        self.dsep14 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
        self.dsep22 = _RDSepBlock(16, 16)
        self.dsep23 = _RDSepBlock(16, 16)
        self.dsep24 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
        self.dsep32 = _RDSepBlock(8, 8)
        self.dsep33 = _RDSepBlock(8, 8)
        self.dsep34 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = self.dsep04(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = self.dsep14(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = self.dsep24(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = self.dsep34(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    


class DSepST7Net_4b_4e_6l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST7Net_4b_4e_6l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RDSepBlock(in_ch, 64)
        self.dsep01 = _RDSepBlock(64, 64)
        self.dsep02 = _RDSepBlock(64, 64)
        self.dsep03 = _RDSepBlock(64, 64)
        self.dsep04 = _RDSepBlock(64, 64)
        self.dsep05 = _RDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        
        
        self.dsep10 = _RDSepBlock(32, 32)
        self.dsep11 = _RDSepBlock(32, 32)
        self.dsep12 = _RDSepBlock(32, 32)
        self.dsep13 = _RDSepBlock(32, 32)
        self.dsep14 = _RDSepBlock(32, 32)
        self.dsep15 = _RDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RDSepBlock(16, 16)
        self.dsep21 = _RDSepBlock(16, 16)
        self.dsep22 = _RDSepBlock(16, 16)
        self.dsep23 = _RDSepBlock(16, 16)
        self.dsep24 = _RDSepBlock(16, 16)
        self.dsep25 = _RDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RDSepBlock(8, 8)
        self.dsep31 = _RDSepBlock(8, 8)
        self.dsep32 = _RDSepBlock(8, 8)
        self.dsep33 = _RDSepBlock(8, 8)
        self.dsep34 = _RDSepBlock(8, 8)
        self.dsep35 = _RDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = self.dsep04(x)
        x = self.dsep05(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = self.dsep14(x)
        x = self.dsep15(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = self.dsep24(x)
        x = self.dsep25(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = self.dsep34(x)
        x = self.dsep35(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x    



class DSepST15Net_4b_4e_1l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_4b_4e_1l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
#         self.dsep01 = _RVarDSepBlock(64, 64)
#         self.dsep02 = _RVarDSepBlock(64, 64)
#         self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
#         self.dsep11 = _RVarDSepBlock(32, 32)
#         self.dsep12 = _RVarDSepBlock(32, 32)
#         self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
#         self.dsep21 = _RVarDSepBlock(16, 16)
#         self.dsep22 = _RVarDSepBlock(16, 16)
#         self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
#         self.dsep31 = _RVarDSepBlock(8, 8)
#         self.dsep32 = _RVarDSepBlock(8, 8)
#         self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
#         x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
#         x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
#         x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
#         x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

class DSepST15Net_4b_4e_2l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_4b_4e_2l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
#         self.dsep02 = _RVarDSepBlock(64, 64)
#         self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
#         self.dsep12 = _RVarDSepBlock(32, 32)
#         self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
#         self.dsep22 = _RVarDSepBlock(16, 16)
#         self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
#         self.dsep32 = _RVarDSepBlock(8, 8)
#         self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
#         x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
#         x = self.dsep12(x)
#         x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
#         x = self.dsep22(x)
#         x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
#         x = self.dsep32(x)
#         x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x



    

class DSepST15Net_4b_4e_3l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_4b_4e_3l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
#         self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
#         self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
#         self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
#         self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
#         x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
#         x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
#         x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
#         x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    

    

    

class DSepST15Net_4b_4e_5l(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_4b_4e_5l, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.dsep04 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.dsep14 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.dsep24 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.dsep34 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        self.convbranch_1 = torch.nn.Conv1d(winsize, hidden_size2, kernel_size=1, padding=0)
        self.conv = torch.nn.Conv1d(dsep_out_ch, hidden_size2, kernel_size=1, padding=0)
        
        self.dense1 = nn.Linear(winsize * hidden_size2 + dsep_out_ch * hidden_size2, 8)
        self.dense2 = nn.Linear(8, 1)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=8)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = self.dsep04(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = self.dsep14(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = self.dsep24(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = self.dsep34(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
        x = x.permute(0, 2, 1)

        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

    


class DSepST15Net_no_branch(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super(DSepST15Net_no_branch, self).__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        dsep_out_ch = 4
        
        
        flattened_size = 240

        self.dense1 = nn.Linear(flattened_size, 240)
        self.dense2 = nn.Linear(240, 1)

        
        self.relu = torch.nn.ReLU()
 
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size2)
        
        self.bn_dense_1 = torch.nn.BatchNorm1d(num_features=240)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
                
        x = self.dsep00(x)
        x = self.dsep01(x)
        x = self.dsep02(x)
        x = self.dsep03(x)
        x = x.permute(0, 2, 1)
        x = self.avg0(x)
        x = x.permute(0, 2, 1)

        x = self.dsep10(x)
        x = self.dsep11(x)
        x = self.dsep12(x)
        x = self.dsep13(x)
        x = x.permute(0, 2, 1)
        x = self.avg1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep20(x)
        x = self.dsep21(x)
        x = self.dsep22(x)
        x = self.dsep23(x)
        x = x.permute(0, 2, 1)
        x = self.avg2(x)
        x = x.permute(0, 2, 1)
        
        x = self.dsep30(x)
        x = self.dsep31(x)
        x = self.dsep32(x)
        x = self.dsep33(x)
        x = x.permute(0, 2, 1)
        x = self.avg3(x)
#         print('self.avg3(x)', x.shape)
        x = x.permute(0, 2, 1) 
#         print('x.permute', x.shape)

        x = torch.flatten(x, start_dim=1) 
        
        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        x = torch.squeeze(x)
    
        return x

