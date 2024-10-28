import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import *


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

# for before dsepst11net
# class LSTM(nn.LSTM):
#     def __init__(self, *args, dropouti: float=0.,
#                  dropoutw: float=0., dropouto: float=0.,
#                  batch_first=True, unit_forget_bias=True, **kwargs):
#         super().__init__(*args, **kwargs, batch_first=batch_first)
#         self.unit_forget_bias = unit_forget_bias
#         self.dropoutw = dropoutw
#         self.input_drop = VariationalDropout(dropouti,
#                                              batch_first=batch_first)
#         self.output_drop = VariationalDropout(dropouto,
#                                               batch_first=batch_first)
#         self._init_weights()

#     def _init_weights(self):
#         """
#         Use orthogonal init for recurrent layers, xavier uniform for input layers
#         Bias is 0 except for forget gate
#         """
#         for name, param in self.named_parameters():
#             if "weight_hh" in name:
#                 nn.init.orthogonal_(param.data)
#             elif "weight_ih" in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif "bias" in name and self.unit_forget_bias:
#                 nn.init.zeros_(param.data)
#                 param.data[self.hidden_size:2 * self.hidden_size] = 1

#     def _drop_weights(self):
#         for name, param in self.named_parameters():
#             if "weight_hh" in name:
#                 getattr(self, name).data = \
#                     torch.nn.functional.dropout(param.data, p=self.dropoutw,
#                                                 training=self.training).contiguous()

#     def forward(self, input, hx=None):
#         self._drop_weights()
#         input = self.input_drop(input)
#         seq, state = super().forward(input, hx=hx)
#         return self.output_drop(seq), state


# for from dsepst11net on
class LSTM(nn.Module):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__()
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self.lstm = _LSTM(*args, **kwargs, batch_first=batch_first)

    def forward(self, input, hx=None):
        input = self.input_drop(input)
        seq, state = self.lstm(input, hx=hx)
        return self.output_drop(seq), state


class _LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        seq, state = super().forward(input, hx=hx)
        return seq, state

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



class SingleAIOSA(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(SingleAIOSA, self).__init__()

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

        self.fc_1 = nn.Linear(256, 60)
        self.fc_2 = nn.Linear(60, 1)


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
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = torch.squeeze(x)
        return x


# NO DROP BOTH CONV_2d and LSTM dropoout    11-12-66 ################################################################
class AIOSANODROP(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(AIOSANODROP, self).__init__()

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
      
        
        self.conv_2 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=2,
                                     padding=int(2*(kernel_size_var-1)/2))
        self.bn_2 = nn.BatchNorm1d(filter_size)



        self.conv_3 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=4,
                                     padding=int(4*(kernel_size_var-1)/2))
        self.bn_3 = nn.BatchNorm1d(filter_size)
     

        self.conv_4 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=8,
                                     padding=int(8*(kernel_size_var-1)/2))
        self.bn_4 = nn.BatchNorm1d(filter_size)


        self.avgPool_a = nn.AvgPool1d(kernel_size=4)



        # Drop 0.1
        self.conv_5 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_5 = nn.BatchNorm1d(filter_size)
    
        self.conv_6 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_6 = nn.BatchNorm1d(filter_size)
  

        self.conv_7 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_7 = nn.BatchNorm1d(filter_size)
      

        self.conv_8 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_8 = nn.BatchNorm1d(filter_size)
    

        self.avgPool_b = nn.AvgPool1d(kernel_size=4)

        
        # 0.3
        self.conv_9 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_9 = nn.BatchNorm1d(filter_size)
  

        self.conv_10 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_10 = nn.BatchNorm1d(filter_size)
     

        self.conv_11 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_11 = nn.BatchNorm1d(filter_size)
 

        self.conv_12 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_12 = nn.BatchNorm1d(filter_size)
           
        

        self.lstm_1 = LSTM(input_size=filter_size, hidden_size=128, num_layers=1,
                           bidirectional=True, batch_first=True, dropouti=0.0)

        self.fc_1 = nn.Linear(256, 1024)
        self.drop_fc_1 = nn.Dropout(0.5)

        self.ln = nn.LayerNorm(1024)
        self.fc_2 = nn.Linear(1024, 60)
        self.fc_3 = nn.Linear(60, 1)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        skip_conn = self.skip(x)


        x = F.relu(self.bn_1(self.conv_1(x)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_2(self.conv_2(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_3(self.conv_3(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_4(self.conv_4(skip_conn)))
        skip_conn = skip_conn.add(x)

        skip_conn = self.avgPool_a(skip_conn)


        x = F.relu(self.bn_5(self.conv_5(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_6(self.conv_6(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_7(self.conv_7(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_8(self.conv_8(skip_conn)))
        skip_conn = skip_conn.add(x)

        skip_conn = self.avgPool_b(skip_conn)


        x = F.relu(self.bn_9(self.conv_9(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_10(self.conv_10(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_11(self.conv_11(skip_conn)))
        skip_conn = skip_conn.add(x)

        x = F.relu(self.bn_12(self.conv_12(skip_conn)))
        skip_conn = skip_conn.add(x)
        

        x = x.permute(0, 2, 1)
        x, states = self.lstm_1(x)
        x = x[:, -1, :]
        x = self.drop_fc_1(F.relu(self.ln(self.fc_1(x))))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = torch.squeeze(x)
        return x

# ################## End  NOOOO DROP BOTH CONV_2d and LSTM dropoout    11-12-66 ############################################


# ################## LSTM variational dropout replace conv_2d  at base conv 12-12-66 ########################################


class AIOSA_LSTM_VD(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(AIOSA_LSTM_VD, self).__init__()

        filter_size = 16
        kernel_size_var = 3


        self.skip = TCSConv1d(in_channels=num_channels, out_channels=filter_size, kernel_size=1,
                              dilation=1, padding=int((1-1)/2))

        self.conv_1 = TCSConv1d(in_channels=num_channels, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_1 = nn.BatchNorm1d(filter_size)
        
        self.conv_2 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=2,
                                     padding=int(2*(kernel_size_var-1)/2))
        self.bn_2 = nn.BatchNorm1d(filter_size)

        self.conv_3 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=4,
                                     padding=int(4*(kernel_size_var-1)/2))
        self.bn_3 = nn.BatchNorm1d(filter_size)
      
        

        self.conv_4 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=8,
                                     padding=int(8*(kernel_size_var-1)/2))
        self.bn_4 = nn.BatchNorm1d(filter_size)
      
        self.avgPool_a = nn.AvgPool1d(kernel_size=4)


        
        self.conv_5 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_5 = nn.BatchNorm1d(filter_size)
    
        self.conv_6 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_6 = nn.BatchNorm1d(filter_size)
  

        self.conv_7 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_7 = nn.BatchNorm1d(filter_size)
      

        self.conv_8 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_8 = nn.BatchNorm1d(filter_size)
        self.avgPool_b = nn.AvgPool1d(kernel_size=4)

        
        
        self.conv_9 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=1,
                                     padding=int((kernel_size_var-1)/2))
        self.bn_9 = nn.BatchNorm1d(filter_size)
  

        self.conv_10 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=3,
                                     padding=int(3*(kernel_size_var-1)/2))
        self.bn_10 = nn.BatchNorm1d(filter_size)
     

        self.conv_11 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=5,
                                     padding=int(5*(kernel_size_var-1)/2))
        self.bn_11 = nn.BatchNorm1d(filter_size)
 

        self.conv_12 = TCSConv1d(in_channels=filter_size, out_channels=filter_size,
                                     kernel_size=kernel_size_var, dilation=9,
                                     padding=int(9*(kernel_size_var-1)/2))
        self.bn_12 = nn.BatchNorm1d(filter_size)
        
        
        self.lstm_0 = LSTM(input_size=filter_size, hidden_size=128, num_layers=1,
                           bidirectional=True, batch_first=True, dropouti=0.1)
        

        self.lstm_1 = LSTM(input_size=filter_size, hidden_size=128, num_layers=1,
                           bidirectional=True, batch_first=True, dropouti=0.0)

        self.fc_1 = nn.Linear(256, 1024)
        self.drop_fc_1 = nn.Dropout(0.5)

        self.ln = nn.LayerNorm(1024)
        self.fc_2 = nn.Linear(1024, 60)
        self.fc_3 = nn.Linear(60, 1)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        skip_conn = self.skip(x)

        x = F.relu(self.bn_1(self.conv_1(x))) 
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
      
    
        x = F.relu(self.bn_2(self.conv_2(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
        
        x = F.relu(self.bn_3(self.conv_3(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
        
        

        x = F.relu(self.bn_4(self.conv_4(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
        skip_conn = self.avgPool_a(skip_conn)


        x = F.relu(self.bn_5(self.conv_5(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct

        x = F.relu(self.bn_6(self.conv_6(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct

        x = F.relu(self.bn_7(self.conv_7(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct

        x = F.relu(self.bn_8(self.conv_8(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
        skip_conn = self.avgPool_b(skip_conn)


        x = F.relu(self.bn_9(self.conv_9(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
       

        x = F.relu(self.bn_10(self.conv_10(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct
      

        x = F.relu(self.bn_11(self.conv_11(skip_conn)))
        skip_conn = skip_conn.add(x)
        x = x.permute(0, 2, 1)   #correct
        x, states = self.lstm_0(x) #correct
        x = x[:, -1, :] #correct


        x = F.relu(self.bn_12(self.conv_12(skip_conn)))
        skip_conn = skip_conn.add(x)
#         x = x.permute(0, 2, 1)   #correct
#         x, states = self.lstm_0(x) #correct
#         x = x[:, -1, :] #correct
        

        x = x.permute(0, 2, 1)
        x, states = self.lstm_1(x)
        x = x[:, -1, :]
        x = self.drop_fc_1(F.relu(self.ln(self.fc_1(x))))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = torch.squeeze(x)
        return x    

# ################## End LSTM variational dropout replace conv_2d  at base conv 12-12-66 ########################################




# Helper function that is used to initialize the weights of the model
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        if "fc_5" in str(m):
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)
        
 

 # MODIFICATION

class AIOSAST(nn.Module):
    def __init__(self, num_channels=7, winsize=None):
        super(AIOSAST, self).__init__()

        filter_size = 16
        kernel_size_var = 3


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


        self.conv_upper = nn.Conv1d(3, 64, kernel_size=1, padding=0)
        self.conv_lower = nn.Conv1d(filter_size, 64, kernel_size=1, padding=0)
        self.bn_lower = nn.BatchNorm1d(num_features=64)
        self.bn_upper = nn.BatchNorm1d(num_features=64)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(3 * 64 * 2, 60)
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
        
        # Lower branch (permutation)
        x = x.permute(0, 2, 1)
        y = self.conv_upper(x)
        y = self.bn_upper(y)
        y = self.relu(y)

        # Upper branch (no permutation)
        x = x.permute(0, 2, 1)
        x = self.conv_lower(x)
        x = self.bn_lower(x)
        x = self.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(x, start_dim=1)
        x = torch.cat([x, y], axis=1)

#         x = self.drop_fc_1(F.relu(self.ln(self.fc_1(x))))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = torch.squeeze(x)
        return x

    
