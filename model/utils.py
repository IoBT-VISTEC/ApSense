import os
import torch
import warnings

SUPPORTED_DATASETS = ['mesa', 'heartbeat']
DL_MODELS = [
    'MiniRRWaveNet', 'RRWaveNet',
    'EEGNetSA',
    'DSepEmbedder',
    'AIOSA', 'AIOSAST', 'AIOSANODROP', 'AIOSA_LSTM_VD', 'AIOSA_branch', 'AIOSA_branch_multi', 'SingleAIOSA',
    'DSepNet', 'DSepNetSmall', 'DSepNetTiny',
    'DSepSTNet', 'DSepST2Net', 'DSepST3Net', 'DSepST4Net',
    'DSepST5Net', 'DSepST6Net', 'DSepST7Net', 'DSepST8Net',
    'DSepST9Net', 'DSepST10Net', 'DSepST11Net', 'DSepST12Net',
    'DSepST13Net', 'DSepST14Net', 'DSepST15Net',  'DSepST16Net_skip', 'DSepST15Net_skip_all_blocks3',
    'DSepST20Net_skip', 'DSepST15Net_skip', 'DSepST7Net_skip_aiosa', 'DSepST7Net_skip_some',
    'DSepST7Net_spatial_dropout',
    'DSepST7Net_spatial_lstm_dropout', 'DSepST15Net_add_no_drop_lstm_final', 'DSepST7Net_spatial_add_no_drop_lstm_final',
    'DSepST7Net_b1', 'DSepST7Net_b2', 'DSepST15Net_add_drop_lstm_final',
    'DSepST7Net_b3', 'DSepST7Net_b5', 'DSepST7Net_4b_4e_1l', 'DSepST7Net_4b_4e_2l', 'DSepST7Net_4b_4e_3l', 'DSepST7Net_4b_4e_5l',
    'DSepST7Net_4b_4e_6l', 'DSepST15Net_4b_4e_1l', 'DSepST15Net_4b_4e_2l', 'DSepST15Net_4b_4e_3l', 'DSepST15Net_4b_4e_5l', 'DSepST15Net_no_branch',
    'ModifiedLeNet5'
]
SUPPORTED_MODELS = DL_MODELS + ['RF', 'XGB', 'SVC']
EVAL_MODES = ['onset', 'severity']

# ## Argument Validation ###


def check_args(args):
    DL_FLAG = args.model in DL_MODELS

    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"The dataset {args.dataset} is currently not supported. Valid values are {SUPPORTED_DATASETS}.")

    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"The model {args.model} is currently not supported. Valid values are {SUPPORTED_MODELS}.")

    if hasattr(args, "finetune_size"):
        args.finetune_size = int(args.finetune_size)

    if not os.path.exists(args.log_dir):
        raise FileNotFoundError(
            f"{args.log_dir} does not exist yet. Please create the directory before running.")

    if not os.path.exists(args.weight_dir):
        raise FileNotFoundError(
            f"{args.weight_dir} does not exist yet. Please create the directory before running.")

    if not args.warning:
        print("==> Suppressing all warnings")
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    if hasattr(args, 'mode') and args.mode not in EVAL_MODES:
        raise ValueError(f"Only {EVAL_MODES} evaluation modes are supported.")

    print("==> All arguments are valid.")

    return DL_FLAG

# ## GPU Configuration ###


def config_gpu(args):
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}.')
    print(f'Using {torch.cuda.device_count()} GPUs.')
    print("==> Finished configuring GPUs.")

    return device
