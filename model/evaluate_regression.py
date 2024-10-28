from argparse import ArgumentParser

import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.svm import SVR
from xgboost import XGBRegressor as XGB

from runner_regressor import load_dataset, evaluate, forward_ml, forward, gen_result
from models import *
from utils import check_args, config_gpu

### Argument Parsing ###
arg_parser = ArgumentParser(description="ApSens Benchmark")

# Experiment
arg_parser.add_argument('--dataset', required=True, help='Dataset for evaluation')
arg_parser.add_argument('--model', required=True, help='Model to be used')
arg_parser.add_argument('--num_folds', default=5, help='Number of data folds (should be pre-processed)')
arg_parser.add_argument('--save_pred', default=None, help='Path to saved predictions')

# Logging directories
arg_parser.add_argument('--dataset_dir', required=True, help='Dataset directory')
arg_parser.add_argument('--log_dir', default='./results', help='Directory to save logs')
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
win_size = 60
wavenet_ch = 7

### Start Running
print("==> Starting the evaluation.")
maes = []
mses = []
r2s = []
infer_times = []

for FOLD in range(int(args.num_folds)):
    train_x, train_y, test_x, test_y = load_dataset(args.dataset, fold=FOLD, data_dir=args.dataset_dir)
    
    if args.save_pred:
        print("==> Using saved predictions.")
        result = np.load(f"{args.save_pred}/pred_{args.dataset}_{args.model}_{FOLD}.npz", allow_pickle=True)
        infer_time = result['infer_time']
        test_pred = result['pred']
        test_pred = np.hstack(test_pred)
    else:
        if DL_FLAG:
            test_pred, infer_time = forward(
                train_set=(train_x, train_y),
                test_set=(test_x, test_y),
                model_class=globals()[args.model],     # refer to "from models import *"
                model_name=args.model,                 # for naming purpose
                dataset_name=args.dataset,             # for naming purpose
                outer_fold=FOLD,
                log_dir=args.log_dir,                  # save your plots
                weight_dir=args.weight_dir,            # save your plots
                device=device
            )

        else:
            test_pred, infer_time = forward_ml(
                test_set=(test_x, test_y),
                model_class=globals()[args.model],     # refer to sklearn and xgboost
                model_name=args.model,                 # for naming purpose
                dataset_name=args.dataset,             # for naming purpose
                outer_fold=FOLD,
                log_dir=args.log_dir,                  # save your plots
                weight_dir=args.weight_dir, 
            )
    
    mae, mse, r2 = evaluate(test_y, test_pred, args, FOLD)
    
    maes.append(mae)
    mses.append(mse)
    r2s.append(r2)
    infer_times.append(infer_time / sum([len(a) for a in test_y]))

print()
mae_msg = gen_result(maes, "MAE", factor=1)
mse_msg = gen_result(mses, "MSE", factor=1)
r2_msg = gen_result(r2, "R2 score", factor=1)
inf_msg = gen_result(infer_times, "Inference time per sample (ms)", factor=10**3)
result = '\n'.join(["-" * 50, mae_msg, mse_msg, r2_msg, inf_msg, "-" * 50])
print(result)

with open(args.log_dir + f'/{args.dataset}_{args.model}.txt', 'w') as f:
    f.write(result)
        
