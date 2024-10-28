import os
from argparse import ArgumentParser

import torch
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.svm import SVR
from xgboost import XGBRegressor as XGB

from runner_regressor import load_dataset, run, run_ml
from models import *
from utils import check_args, config_gpu
from grid_params import get_params

### Argument Parsing ###
arg_parser = ArgumentParser(description="ApSens Benchmark")

# Experiment
arg_parser.add_argument('--dataset', required=True, help='Dataset for evaluation')
arg_parser.add_argument('--model', required=True, help='Model to be used')
arg_parser.add_argument('--grid_params', default=None, help='Hyperparameters for GridSearchCV (only ML mode)')
arg_parser.add_argument('--num_folds', type=int, default=5, help='Number of data folds (should be pre-processed)')
arg_parser.add_argument('--starting_fold', type=int, default=0, help='The starting data fold')

# Logging directories
arg_parser.add_argument('--dataset_dir', required=True, help='Dataset directory')
arg_parser.add_argument('--log_dir', default='./logs', help='Directory to save logs')
arg_parser.add_argument('--weight_dir', default='./weights', help='Directory to save model weights')

# Device settings
arg_parser.add_argument('--gpu', default=None, help='GPU devices for setting "CUDA_VISIBLE_DEVICES"')

# Warnings
arg_parser.add_argument('--warning', action='store_true', help='Show warning')

print("==> Initializing.")

### Argument Validation ###
args = arg_parser.parse_args()
DL_FLAG = check_args(args)

### GPU Configuration ###
device = config_gpu(args) if DL_FLAG else 'cpu'

# TODO: Import by files
lr = 1e-4
batch_size = 64
total_epoches = 10000
# n_epochs_stop = 30
win_size = 60
wavenet_ch = 7

import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print(time_string)

### Start Running
print("==> Starting the training.")
for FOLD in range(args.starting_fold, args.starting_fold + args.num_folds):
    all_x, all_y, test_x, test_y = load_dataset(args.dataset, fold=FOLD, data_dir=args.dataset_dir)
    
    if DL_FLAG:
        run(
            train_set=(all_x, all_y),
            test_set=(test_x, test_y),
            model_class=globals()[args.model],     # refer to "from models import *"
            model_name=args.model,                 # for naming purpose
            dataset_name=args.dataset,             # for naming purpose
            outer_fold=FOLD,
            log_dir=args.log_dir,                  # save your plots
            weight_dir=args.weight_dir,            # save your plots
            device=device,
            save_pred=True,

            lr=lr,
            batch_size=batch_size
        )
    
    else:
        grid_params = get_params(args.model)
        run_ml(
            train_set=(all_x, all_y),
            test_set=(test_x, test_y),
            model_class=globals()[args.model],     # refer to sklearn and xgboost
            model_name=args.model,                 # for naming purpose
            dataset_name=args.dataset,             # for naming purpose
            outer_fold=FOLD,
            log_dir=args.log_dir,                  # save your plots
            weight_dir=args.weight_dir, 
            grid_params=grid_params,
            save_pred=True,
        )
        
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print(time_string)