import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import *
from .aiosa import LSTM


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

# Separable Convs in Pytorch
# https://gist.github.com/iiSeymour/85a5285e00cbed60537241da7c3b8525

class TCSConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TCSConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, dilation=dilation, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class AIOSA(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(AIOSA, self).__init__()

        filter_size = 16
        kernel_size_var = 3

        # W:input volume size
        # F:kernel size
        # S:stride
        # P:amount of padding
        # size of output volume = (W-F+2P)/S+1

        # to keep the same size, padding = dilation * (kernel - 1) / 2


        self.skip = TCSConv1d(in_channels=num_channels, out_channels=filter_size, kernel_size=1,
                              dilation=1, padding=int((1-1)/2))


        # Drop 0.2

        self.conv_1 = TCSConv1d(in_channels=num_channels, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_1 = nn.BatchNorm1d(filter_size)
        self.drop_1 = nn.Dropout2d(0.2)
        
        self.conv_2 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=2,
                                     padding=int(2*(kernel_size_var-1)/2))
        self.bn_2 = nn.BatchNorm1d(filter_size)
        self.drop_2 = nn.Dropout2d(0.2)


        self.conv_3 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=4,
                                     padding=int(4*(kernel_size_var-1)/2))
        self.bn_3 = nn.BatchNorm1d(filter_size)
        self.drop_3 = nn.Dropout2d(0.2)


        self.conv_4 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=8,
                                     padding=int(8*(kernel_size_var-1)/2))
        self.bn_4 = nn.BatchNorm1d(filter_size)
        self.drop_4 = nn.Dropout2d(0.2)

        self.avgPool_a = nn.AvgPool1d(kernel_size=4)



        # Drop 0.1
        self.conv_5 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_5 = nn.BatchNorm1d(filter_size)
        self.drop_5 = nn.Dropout2d(0.1)

        self.conv_6 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_6 = nn.BatchNorm1d(filter_size)
        self.drop_6 = nn.Dropout2d(0.1)

        self.conv_7 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_7 = nn.BatchNorm1d(filter_size)
        self.drop_7 = nn.Dropout2d(0.1)

        self.conv_8 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_8 = nn.BatchNorm1d(filter_size)
        self.drop_8 = nn.Dropout2d(0.1)

        self.avgPool_b = nn.AvgPool1d(kernel_size=4)

        # 0.3
        self.conv_9 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_9 = nn.BatchNorm1d(filter_size)
        self.drop_9 = nn.Dropout2d(0.3)

        self.conv_10 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_10 = nn.BatchNorm1d(filter_size)
        self.drop_10 = nn.Dropout2d(0.3)

        self.conv_11 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_11 = nn.BatchNorm1d(filter_size)
        self.drop_11 = nn.Dropout2d(0.3)

        self.conv_12 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_12 = nn.BatchNorm1d(filter_size)
        self.drop_12 = nn.Dropout2d(0.3)


        self.lstm_1 = LSTM(input_size=filter_size, hidden_size=128, num_layers=1,
                           bidirectional=True, batch_first=True, dropouti=0.1)

        self.fc_1 = nn.Linear(256, 1024)
        self.drop_fc_1 = nn.Dropout(0.5)

        self.ln = nn.LayerNorm(1024)
        self.fc_2 = nn.Linear(1024, 60)
        self.fc_3 = nn.Linear(60, 1)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        skip_conn = self.skip(x)


        x = self.drop_1(F.relu(self.bn_1(self.conv_1(x))))
        skip_conn = skip_conn.add(x)

        x = self.drop_2(F.relu(self.bn_2(self.conv_2(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_3(F.relu(self.bn_3(self.conv_3(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_4(F.relu(self.bn_4(self.conv_4(skip_conn))))
        skip_conn = skip_conn.add(x)

        skip_conn = self.avgPool_a(skip_conn)


        x = self.drop_5(F.relu(self.bn_5(self.conv_5(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_6(F.relu(self.bn_6(self.conv_6(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_7(F.relu(self.bn_7(self.conv_7(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_8(F.relu(self.bn_8(self.conv_8(skip_conn))))
        skip_conn = skip_conn.add(x)

        skip_conn = self.avgPool_b(skip_conn)


        x = self.drop_9(F.relu(self.bn_9(self.conv_9(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_10(F.relu(self.bn_10(self.conv_10(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_11(F.relu(self.bn_11(self.conv_11(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_12(F.relu(self.bn_12(self.conv_12(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = x.permute(0, 2, 1)
        x, states = self.lstm_1(x)
        x = x[:, -1, :]
        x = self.drop_fc_1(F.relu(self.ln(self.fc_1(x))))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = torch.squeeze(x)
        return x


# Helper function that is used to initialize the weights of the model
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        if "fc_5" in str(m):
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)

# MODIFICATION



class AIOSA_branch(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(AIOSA_branch, self).__init__()

        filter_size = 16
        kernel_size_var = 3

        # W:input volume size
        # F:kernel size
        # S:stride
        # P:amount of padding
        # size of output volume = (W-F+2P)/S+1

        # to keep the same size, padding = dilation * (kernel - 1) / 2


        self.skip = TCSConv1d(in_channels=num_channels, out_channels=filter_size, kernel_size=1,
                              dilation=1, padding=int((1-1)/2))


        # Drop 0.2

        self.conv_1 = TCSConv1d(in_channels=num_channels, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_1 = nn.BatchNorm1d(filter_size)
        self.drop_1 = nn.Dropout2d(0.2)

        self.conv_2 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=2,
                                     padding=int(2*(kernel_size_var-1)/2))
        self.bn_2 = nn.BatchNorm1d(filter_size)
        self.drop_2 = nn.Dropout2d(0.2)


        self.conv_3 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=4,
                                     padding=int(4*(kernel_size_var-1)/2))
        self.bn_3 = nn.BatchNorm1d(filter_size)
        self.drop_3 = nn.Dropout2d(0.2)


        self.conv_4 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=8,
                                     padding=int(8*(kernel_size_var-1)/2))
        self.bn_4 = nn.BatchNorm1d(filter_size)
        self.drop_4 = nn.Dropout2d(0.2)

        self.avgPool_a = nn.AvgPool1d(kernel_size=4)



        # Drop 0.1
        self.conv_5 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_5 = nn.BatchNorm1d(filter_size)
        self.drop_5 = nn.Dropout2d(0.1)

        self.conv_6 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_6 = nn.BatchNorm1d(filter_size)
        self.drop_6 = nn.Dropout2d(0.1)

        self.conv_7 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_7 = nn.BatchNorm1d(filter_size)
        self.drop_7 = nn.Dropout2d(0.1)

        self.conv_8 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_8 = nn.BatchNorm1d(filter_size)
        self.drop_8 = nn.Dropout2d(0.1)

        self.avgPool_b = nn.AvgPool1d(kernel_size=4)

        # 0.3
        self.conv_9 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_9 = nn.BatchNorm1d(filter_size)
        self.drop_9 = nn.Dropout2d(0.3)

        self.conv_10 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_10 = nn.BatchNorm1d(filter_size)
        self.drop_10 = nn.Dropout2d(0.3)

        self.conv_11 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_11 = nn.BatchNorm1d(filter_size)
        self.drop_11 = nn.Dropout2d(0.3)

        self.conv_12 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_12 = nn.BatchNorm1d(filter_size)
        self.drop_12 = nn.Dropout2d(0.3)

        lstm_hidden_size = 128
        self.lstm_1 = LSTM(input_size=filter_size, hidden_size=lstm_hidden_size, num_layers=1,
                           bidirectional=True, batch_first=True, dropouti=0.1)

        convbranch_hidden_size = 8
        self.convbranch_1 = torch.nn.Conv1d(winsize //4 //4, convbranch_hidden_size, kernel_size=1, padding=0)
        self.bn3 = torch.nn.BatchNorm1d(num_features=convbranch_hidden_size)

        self.fc_1 = nn.Linear(lstm_hidden_size * 2 + convbranch_hidden_size * filter_size, 1024)
        self.drop_fc_1 = nn.Dropout(0.5)

        self.ln = nn.LayerNorm(1024)
        self.fc_2 = nn.Linear(1024, 60)
        self.fc_3 = nn.Linear(60, 1)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        
        skip_conn = self.skip(x)


        x = self.drop_1(F.relu(self.bn_1(self.conv_1(x))))
        skip_conn = skip_conn.add(x)

        x = self.drop_2(F.relu(self.bn_2(self.conv_2(skip_conn))))
        skip_conn = skip_conn.add(x)
        
        x = self.drop_3(F.relu(self.bn_3(self.conv_3(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_4(F.relu(self.bn_4(self.conv_4(skip_conn))))
        skip_conn = skip_conn.add(x)

        skip_conn = self.avgPool_a(skip_conn)

        x = self.drop_5(F.relu(self.bn_5(self.conv_5(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_6(F.relu(self.bn_6(self.conv_6(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_7(F.relu(self.bn_7(self.conv_7(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_8(F.relu(self.bn_8(self.conv_8(skip_conn))))
        skip_conn = skip_conn.add(x)

        skip_conn = self.avgPool_b(skip_conn)


        x = self.drop_9(F.relu(self.bn_9(self.conv_9(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_10(F.relu(self.bn_10(self.conv_10(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_11(F.relu(self.bn_11(self.conv_11(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_12(F.relu(self.bn_12(self.conv_12(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = x.permute(0, 2, 1)
        
        y = self.convbranch_1(x)
        y = self.bn3(y)
        y = F.relu(y)

        x, states = self.lstm_1(x)
        x = x[:, -1, :]

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = torch.cat([x, y], 1)
        x = self.drop_fc_1(F.relu(self.ln(self.fc_1(x))))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = torch.squeeze(x)
        return x

class AIOSA_branch_multi(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(AIOSA_branch_multi, self).__init__()

        filter_size = 16
        kernel_size_var = 3

        # W:input volume size
        # F:kernel size
        # S:stride
        # P:amount of padding
        # size of output volume = (W-F+2P)/S+1

        # to keep the same size, padding = dilation * (kernel - 1) / 2


        self.skip = TCSConv1d(in_channels=num_channels, out_channels=filter_size, kernel_size=1,
                              dilation=1, padding=int((1-1)/2))


        # Drop 0.2

        self.conv_1 = TCSConv1d(in_channels=num_channels, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_1 = nn.BatchNorm1d(filter_size)
        self.drop_1 = nn.Dropout2d(0.2)

        self.conv_2 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=2,
                                     padding=int(2*(kernel_size_var-1)/2))
        self.bn_2 = nn.BatchNorm1d(filter_size)
        self.drop_2 = nn.Dropout2d(0.2)


        self.conv_3 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=4,
                                     padding=int(4*(kernel_size_var-1)/2))
        self.bn_3 = nn.BatchNorm1d(filter_size)
        self.drop_3 = nn.Dropout2d(0.2)


        self.conv_4 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=8,
                                     padding=int(8*(kernel_size_var-1)/2))
        self.bn_4 = nn.BatchNorm1d(filter_size)
        self.drop_4 = nn.Dropout2d(0.2)

        self.avgPool_a = nn.AvgPool1d(kernel_size=4)



        # Drop 0.1
        self.conv_5 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_5 = nn.BatchNorm1d(filter_size)
        self.drop_5 = nn.Dropout2d(0.1)

        self.conv_6 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_6 = nn.BatchNorm1d(filter_size)
        self.drop_6 = nn.Dropout2d(0.1)

        self.conv_7 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_7 = nn.BatchNorm1d(filter_size)
        self.drop_7 = nn.Dropout2d(0.1)

        self.conv_8 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_8 = nn.BatchNorm1d(filter_size)
        self.drop_8 = nn.Dropout2d(0.1)

        self.avgPool_b = nn.AvgPool1d(kernel_size=4)

        # 0.3
        self.conv_9 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_9 = nn.BatchNorm1d(filter_size)
        self.drop_9 = nn.Dropout2d(0.3)

        self.conv_10 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_10 = nn.BatchNorm1d(filter_size)
        self.drop_10 = nn.Dropout2d(0.3)

        self.conv_11 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_11 = nn.BatchNorm1d(filter_size)
        self.drop_11 = nn.Dropout2d(0.3)

        self.conv_12 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_12 = nn.BatchNorm1d(filter_size)
        self.drop_12 = nn.Dropout2d(0.3)

        lstm_hidden_size = 128
        self.lstm_1 = LSTM(input_size=filter_size, hidden_size=lstm_hidden_size, num_layers=1,
                           bidirectional=True, batch_first=True, dropouti=0.1)

        convbranch_hidden_size = 8
        self.convbranch_1 = torch.nn.Conv1d(winsize, convbranch_hidden_size, kernel_size=1, padding=0)
        self.bn_branch_1 = torch.nn.BatchNorm1d(num_features=convbranch_hidden_size)

        self.convbranch_2 = torch.nn.Conv1d(winsize //4, convbranch_hidden_size, kernel_size=1, padding=0)
        self.bn_branch_2 = torch.nn.BatchNorm1d(num_features=convbranch_hidden_size)

        self.convbranch_3 = torch.nn.Conv1d(winsize //4 //4, convbranch_hidden_size, kernel_size=1, padding=0)
        self.bn_branch_3 = torch.nn.BatchNorm1d(num_features=convbranch_hidden_size)

        self.fc_1 = nn.Linear(lstm_hidden_size * 2 + convbranch_hidden_size * filter_size * 3, 1024)
        self.drop_fc_1 = nn.Dropout(0.5)

        self.ln = nn.LayerNorm(1024)
        self.fc_2 = nn.Linear(1024, 60)
        self.fc_3 = nn.Linear(60, 1)


    def forward(self, x):
        #print(x.shape)
        x = torch.transpose(x, 1, 2)

        #print(x.shape)

        skip_conn = self.skip(x)


        x = self.drop_1(F.relu(self.bn_1(self.conv_1(x))))
        skip_conn = skip_conn.add(x)

        x = self.drop_2(F.relu(self.bn_2(self.conv_2(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_3(F.relu(self.bn_3(self.conv_3(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_4(F.relu(self.bn_4(self.conv_4(skip_conn))))
        skip_conn = skip_conn.add(x)

        y = F.relu(self.bn_branch_1(self.convbranch_1(x.permute(0, 2, 1))))

        skip_conn = self.avgPool_a(skip_conn)

        x = self.drop_5(F.relu(self.bn_5(self.conv_5(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_6(F.relu(self.bn_6(self.conv_6(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_7(F.relu(self.bn_7(self.conv_7(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_8(F.relu(self.bn_8(self.conv_8(skip_conn))))
        skip_conn = skip_conn.add(x)

        z = F.relu(self.bn_branch_2(self.convbranch_2(x.permute(0, 2, 1))))

        skip_conn = self.avgPool_b(skip_conn)


        x = self.drop_9(F.relu(self.bn_9(self.conv_9(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_10(F.relu(self.bn_10(self.conv_10(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_11(F.relu(self.bn_11(self.conv_11(skip_conn))))
        skip_conn = skip_conn.add(x)

        x = self.drop_12(F.relu(self.bn_12(self.conv_12(skip_conn))))
        skip_conn = skip_conn.add(x)

        w = F.relu(self.bn_branch_3(self.convbranch_3(x.permute(0, 2, 1))))

        x = x.permute(0, 2, 1)

        #print(x.shape)
        #print(skip_conn.shape)

        #print(x.shape)
        x, states = self.lstm_1(x)
        x = x[:, -1, :]

        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        y = torch.flatten(y, start_dim=1)
        z = torch.flatten(z, start_dim=1)
        w = torch.flatten(w, start_dim=1)
        #print(y.shape)
        x = torch.cat([x, y, z, w], 1)
        #print(x.shape)
        x = self.drop_fc_1(F.relu(self.ln(self.fc_1(x))))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = torch.squeeze(x)
        return x