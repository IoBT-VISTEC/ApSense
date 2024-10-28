from .dsepembedder import DSepEmbedder
from .dsepnet import (
    DSepNet, DSepNetSmall, DSepNetTiny,
    DSepSTNet, DSepST2Net, DSepST3Net, DSepST4Net,
    DSepST5Net, DSepST6Net, DSepST7Net, DSepST8Net,
    DSepST9Net, DSepST10Net, DSepST11Net, DSepST12Net,
    DSepST13Net, DSepST14Net, DSepST15Net, DSepST16Net_skip, DSepST15Net_skip_all_blocks3, DSepST7Net_skip_some,
    DSepST20Net_skip, DSepST15Net_skip, DSepST7Net_skip_aiosa, DSepST7Net_spatial_dropout, DSepST7Net_spatial_lstm_dropout, DSepST15Net_add_no_drop_lstm_final, DSepST7Net_spatial_add_no_drop_lstm_final, DSepST15Net_add_drop_lstm_final, DSepST7Net_b1, DSepST7Net_b2, DSepST7Net_b3, DSepST7Net_b5,
    DSepST7Net_4b_4e_1l, DSepST7Net_4b_4e_2l, DSepST7Net_4b_4e_3l, DSepST7Net_4b_4e_5l, DSepST7Net_4b_4e_6l, DSepST15Net_4b_4e_1l, DSepST15Net_4b_4e_2l, DSepST15Net_4b_4e_3l, DSepST15Net_4b_4e_5l, DSepST15Net_no_branch
)
from .mini_rrwavenet import MiniRRWaveNet
from .rrwavenet import RRWaveNet
from .eeg_net_sa import EEGNetSA
from .aiosa import AIOSA, AIOSAST, AIOSANODROP, AIOSA_LSTM_VD, SingleAIOSA
from .aiosa_modify import AIOSA_branch, AIOSA_branch_multi
from .lenet5 import ModifiedLeNet5

MODELS = {
    "DSepEmbedder": DSepEmbedder,
    "DSepNet": DSepNet,
    "DSepNetSmall": DSepNetSmall,
    "DSepNetTiny": DSepNetTiny,
    "DSepSTNet": DSepSTNet,
    "DSepST2Net": DSepST2Net,
    "DSepST3Net": DSepST3Net,
    "DSepST4Net": DSepST4Net,
    "DSepST5Net": DSepST5Net,
    "DSepST6Net": DSepST6Net,
    "DSepST7Net": DSepST7Net,
    "DSepST8Net": DSepST8Net,
    "DSepST9Net": DSepST9Net,
    "DSepST10Net": DSepST10Net,
    "DSepST11Net": DSepST11Net,
    "DSepST12Net": DSepST12Net,
    "DSepST13Net": DSepST13Net,
    "DSepST14Net": DSepST14Net,
    "DSepST15Net": DSepST15Net,
    "DSepST15Net_skip": DSepST15Net_skip,
    "DSepST16Net_skip": DSepST16Net_skip,  # Add removed
    "DSepST20Net_skip": DSepST20Net_skip,  # Add
    "DSepST15Net_skip_all_blocks3": DSepST15Net_skip_all_blocks3,  # Add
    "DSepST7Net_skip_aiosa": DSepST7Net_skip_aiosa,  # Add
    "DSepST7Net_skip_some": DSepST7Net_skip_some,
    "DSepST7Net_spatial_dropout": DSepST7Net_spatial_dropout,
    "DSepST7Net_spatial_lstm_dropout": DSepST7Net_spatial_lstm_dropout,
    "DSepST15Net_add_no_drop_lstm_final": DSepST15Net_add_no_drop_lstm_final,
    "DSepST7Net_spatial_add_no_drop_lstm_final": DSepST7Net_spatial_add_no_drop_lstm_final,
    "DSepST15Net_add_drop_lstm_final": DSepST15Net_add_drop_lstm_final,
    "DSepST7Net_b1": DSepST7Net_b1,
    "DSepST7Net_b2": DSepST7Net_b2,
    "DSepST7Net_b3": DSepST7Net_b3,
    "DSepST7Net_b5": DSepST7Net_b5,
    "DSepST7Net_4b_4e_1l": DSepST7Net_4b_4e_1l,
    "DSepST7Net_4b_4e_2l": DSepST7Net_4b_4e_2l,
    "DSepST7Net_4b_4e_3l": DSepST7Net_4b_4e_3l,
    "DSepST7Net_4b_4e_5l": DSepST7Net_4b_4e_5l,
    "DSepST7Net_4b_4e_6l": DSepST7Net_4b_4e_6l,
    "DSepST15Net_4b_4e_1l": DSepST15Net_4b_4e_1l,
    "DSepST15Net_4b_4e_2l": DSepST15Net_4b_4e_2l,
    "DSepST15Net_4b_4e_3l": DSepST15Net_4b_4e_3l,
    "DSepST15Net_4b_4e_5l": DSepST15Net_4b_4e_5l,
    "DSepST15Net_no_branch": DSepST15Net_no_branch,
    "MiniRRWaveNet": MiniRRWaveNet,
    "EEGNetSA": EEGNetSA,
    "AIOSA": AIOSA,
    "AIOSAST": AIOSAST,
    "AIOSANODROP": AIOSANODROP,
    "AIOSA_LSTM_VD":  AIOSA_LSTM_VD,
    "AIOSA_branch": AIOSA_branch,
    "AIOSA_branch_multi": AIOSA_branch_multi,
    "SingleAIOSA": SingleAIOSA,
    "RRWaveNet": RRWaveNet,
    "ModifiedLeNet5": ModifiedLeNet5,
}
