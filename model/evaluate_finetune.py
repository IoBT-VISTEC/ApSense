from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from xgboost import XGBClassifier as XGB

from runner import load_dataset, evaluate_onset, forward_finetune, gen_result, find_ls_r2, bland_altman_plot
from models import *
from utils import check_args, config_gpu

### Argument Parsing ###
arg_parser = ArgumentParser(description="ApSens Benchmark")

# Experiment
arg_parser.add_argument('--dataset', required=True, help='Dataset for evaluation')
arg_parser.add_argument('--model', required=True, help='Model to be used')
arg_parser.add_argument('--num_folds', default=5, help='Number of data folds (should be pre-processed)')
arg_parser.add_argument('--mode', default='onset', help='Evaluation mode: by onset or by severity')
arg_parser.add_argument('--finetune_size', default=10, help='Percent of finetuning samples from the test set')

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
accs = []
f1s = []
sens = []
specs = []

data = {}

for FOLD in range(int(args.num_folds)):
    print("FOLD", FOLD)
    train_x, train_y, test_x, test_y = load_dataset(args.dataset, fold=FOLD, data_dir=args.dataset_dir)
    
    acc, f1, sen, spec, d = forward_finetune(
        test_set=(test_x, test_y),
        model_class=globals()[args.model],     # refer to "from models import *"
        model_name=args.model,                 # for naming purpose
        dataset_name=args.dataset,             # for naming purpose
        outer_fold=FOLD,
        log_dir=args.log_dir,                  # save your plots
        weight_dir=args.weight_dir,            # save your plots
        device=device,
        finetune_size=args.finetune_size,
    )
    print(type(test_x))
    print("Accuracy:", np.mean(acc))
    print("Macro F1:", np.mean(f1))
    accs.append(np.mean(acc))
    f1s.append(np.mean(f1))
    sens.append(sen)
    specs.append(spec)
    
    data[FOLD] = d
    

                       
                     
    
accs = np.hstack(accs)
f1s = np.hstack(f1s)

print()
acc_msg = gen_result(accs, "Accuracy")
f1_msg = gen_result(f1s, "F1")
sens_msg = gen_result(sens, "Sensitivity")
spec_msg = gen_result(specs, "Specificity")
result = '\n'.join(["-" * 50, acc_msg, f1_msg, sens_msg, spec_msg, "-" * 50])
print(result)

with open(args.log_dir + f'/{args.dataset}_{args.model}_finetune{args.finetune_size}.pckl', 'wb') as f:
    pickle.dump(data, f)

with open(args.log_dir + f'/{args.dataset}_{args.model}_finetune{args.finetune_size}.txt', 'w') as f:
    f.write(result)

# with open(args.log_dir + f'/{args.dataset}_{args.model}_finetune{args.finetune_size}_for_res_anlys.pckl', 'wb') as f:
#     pickle.dump(data, f)