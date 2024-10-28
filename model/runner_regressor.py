import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
import time

import torch
from torch import nn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import RandomUnderSampler
from losses import *

# for consistency, all seeds are set to 69420
seed = 69420
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def ends(seq):
    curr = 0
    count = 0
    for item in seq:
        if item != 0:
            curr = 1
        else: # reverts from 1 to 0
            if curr == 1:
                count += 1
                curr = 0
    return count


def make_y(all_y):
    ys = []
    for window in all_y:
        ys.append(ends(window))
    ys = np.array(ys)
    return ys

def scaler_all_channel(train_x=None, test_x=None, scalers=None, ch=-1):
    prescaled = scalers is not None
    
    if (train_x is None and not prescaled) or (train_x is not None and prescaled):
        raise ValueError("Only train_x or scalers is required.")
        
    if test_x is None:
        test_x = train_x
    
    scalers = scalers if scalers else []
    scaled_X = []
    num_ch = train_x.shape[1] if train_x is not None else test_x.shape[1]
    
    if ch == -1: #SCALER ALL CHANNEL
        for i in range(num_ch):
            if prescaled:
                scaler = scalers[i]
            else:
                scaler = StandardScaler()
                scaler.fit(train_x[:, i, :])
                
            scaled_X.append(scaler.transform(test_x[:, i, :]))
            scalers.append(scaler)

    else:
        if prescaled:
            scaler = scalers[i]
        else:
            scaler.fit(train_x[:, ch, :])
            
        scaled_X.append(scaler.transform(test_x[:, ch, :]))
        
    scaled_X = np.array(scaled_X)    
    scaled_X = np.swapaxes(scaled_X, 0, 1)
    scaled_X = np.swapaxes(scaled_X, 1, 2)

    return scaled_X, scalers


def load_dataset(dataset, fold, data_dir='/mount/guntitats/apsens_processed'):
    try:
        x_train_file = data_dir + f'/{dataset}_fold{fold}_x_train.pickle'
        y_train_file = data_dir + f'/{dataset}_fold{fold}_y_train.pickle'
        x_test_file = data_dir + f'/{dataset}_fold{fold}_x_test.pickle'
        y_test_file = data_dir + f'/{dataset}_fold{fold}_y_test.pickle'

        all_x = pd.read_pickle(x_train_file)
        all_y = pd.read_pickle(y_train_file)
        test_x = pd.read_pickle(x_test_file)
        test_y = pd.read_pickle(y_test_file)

        return all_x, all_y, test_x, test_y
    
    except:
        raise FileNotFoundError(f"Some files do not exist, have incorrect format or naming. Please check README.")
        
        
class stdard(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalers = []

    def fit(self, X, y=None):      
        _, self.scalers = scaler_all_channel(X)
        return self

    def transform(self, X, y=None):
        scaled_X = []

        for j in range(X.shape[1]):
            scaled_X.append(self.scalers[j].transform(X[:, j, :]))

        scaled_X = np.array(scaled_X)
        scaled_X = np.swapaxes(scaled_X, 0, 1)
        concat_X = scaled_X.reshape(-1, scaled_X.shape[1] * scaled_X.shape[2])

        return concat_X

    
class regression_scaler():
    def __init__(self):
#         self.scaler = Pipeline([
#             ("shifter", FunctionTransformer(func=lambda x: x+1, inverse_func=lambda x: x-1)),
#             ("scaler", PowerTransformer(method='box-cox')),
#         ])
        
#         self.scaler = PowerTransformer(method='box-cox')
        self.scaler = FunctionTransformer(func=lambda x: np.cbrt(x), inverse_func=lambda x: x ** 3)
    
    def fit(self, x):
        return self.scaler.fit(x)
    
    def fit_transform(self, x):
        return self.scaler.fit_transform(x)
    
    def transform(self, x):
        return self.scaler.transform(x)
    
    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)


def metrics(gt, pd):
    mae = mean_absolute_error(gt, pd)
    mse = mean_squared_error(gt, pd)
    r2 = r2_score(gt, pd)
    return mae, mse, r2


def arrange_save_pred(gt, pred):
    num_samples_per_subj = [len(x) for x in gt]
    acc_subj = 0
    
    pred_arr = []
    
    for num in num_samples_per_subj:
        pred_arr.append(np.array(pred[acc_subj:acc_subj+num]))
        acc_subj += num
        
    return np.array(pred_arr)


def run_ml(
    train_set, test_set, 
    model_class, model_name, 
    outer_fold, dataset_name,
    log_dir, weight_dir,
    grid_params,
    save_pred=False,
    subsampling=False
):
    # Setting up the dataset
    print("Setting up dataset")
    all_x, all_y = train_set
    test_x, test_y = test_set
    
    all_x = np.array([inx for outx in all_x for inx in outx])
    all_y = np.array([iny for outy in all_y for iny in outy])
    new_x_test = np.vstack(test_x)
    new_y_test = np.vstack(test_y)

    print(f'Train {len(all_x)} samples | Test {len(new_x_test)} samples')

    all_y = make_y(all_y)
    new_y_test = make_y(new_y_test)
    
    splits = KFold(n_splits=5, random_state=42, shuffle=True)
    
    args = {}
    # Prevents too-long runs
    if model_name == 'SVR':
        args['max_iter'] = 10000
        
    estimators = [('std', stdard()), ('reduce_dim', PCA(n_components=0.95)), ('clf', model_class(**args))]
    train_pipe = Pipeline(estimators)
    clf = GridSearchCV(train_pipe, grid_params, cv=splits, scoring='neg_mean_squared_error', n_jobs=-1, verbose=10)
    
    clf.fit(all_x, all_y)
    print(f'Best score from grid search: {clf.best_score_}')
    
    start = time.process_time()
    test_pred = clf.best_estimator_.predict(new_x_test)
    inference_time = time.process_time() - start
    
    test_mae, test_mse, test_r2 = metrics(new_y_test, test_pred)

    model_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pickle"
    with open(model_dir, 'wb') as handle:
        pickle.dump(clf.best_estimator_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Best model saved to {model_dir}')
    
    mae_msg = f"MAE: {test_mae}"
    mse_msg = f"MSE: {test_mse}"
    r2_msg = f"R2 score: {test_r2}"
    inf_msg = f"Inference time: {inference_time:.5f} for {len(new_y_test)} samples"
    result = '\n'.join([mae_msg, mse_msg, r2_msg, inf_msg])
    print(result)
    
    if save_pred:
        arranged_pred = arrange_save_pred(test_set[1], test_pred)
        np.savez(log_dir + f'/pred_{dataset_name}_{model_name}_{outer_fold}', pred=arranged_pred, infer_time=inference_time)
    
    with open(log_dir + f'/best_{dataset_name}_{model_name}_{outer_fold}.txt', 'w') as f:
        f.write(result)


def run(
    train_set, test_set, 
    model_class, model_name, 
    outer_fold, dataset_name,
    log_dir, weight_dir,
    device,
    subsampling=False,
    save_pred=False,
    lr=1e-3, batch_size=1024, total_epoches=10000, n_epochs_stop=30, win_size=60, wavenet_ch=7, save_emb_pred = True
):
    all_x, all_y = train_set
    test_x, test_y = test_set
    
    splits = KFold(n_splits=5, random_state=42, shuffle=True)
    
    best_test_mae = []
    best_test_mse = []
    best_mae = 100000
    best_mse = 100000
    
    fold_test_mae = []
    fold_test_mse = []
    fold_test_r2 = []
    
    all_x = np.array([inx for outx in all_x for inx in outx])
    all_y = np.array([iny for outy in all_y for iny in outy])
    
#     all_x = np.vstack([np.array([x[,:300], x[,300:] for x in all_x])
#     all_y = np.vstack([np.array([y[,:300], y[,300:] for x in all_x])

    all_y = make_y(all_y)
#     all_x = all_x[all_y > 0] #
#     all_y = all_y[all_y > 0] #

    new_x_test = torch.FloatTensor(np.vstack(test_x))
    new_y_test = torch.FloatTensor(np.vstack(test_y))
    
    new_y_test = make_y(new_y_test)
    new_y_test = torch.FloatTensor(new_y_test)
#     new_x_test = new_x_test[new_y_test > 0]
#     new_y_test = torch.FloatTensor(new_y_test[new_y_test > 0])
    
    
    for fold, (train_ids, val_ids) in enumerate(splits.split(all_x, all_y)):
        print(f'Running fold {outer_fold}-{fold}')
        print(f'Train {len(train_ids)} samples | Validation {len(val_ids)} samples')

        fold_x_train = all_x[train_ids]
        fold_y_train = all_y[train_ids]
        fold_x_val = all_x[val_ids]
        fold_y_val = all_y[val_ids]
        
        # Trackers
        fold_train_loss, fold_val_loss, fold_val_mae, fold_val_mse, fold_val_r2 = [], [], [], [], []
        epoch_loss, val_epoch_loss, val_epoch_mae, val_epoch_mse, val_epoch_r2 = [], [], [], [], []
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = 10000
        min_val_mse = 100000
        
        # Transform y
#         y_transformer = regression_scaler() #
#         fold_y_train = y_transformer.fit_transform(fold_y_train.reshape(-1, 1)).reshape(-1) #
#         fold_y_val = y_transformer.transform(fold_y_val.reshape(-1, 1)).reshape(-1) #
        
        density, bins = np.histogram(fold_y_train, density=True)
        density = density / np.sum(density)
        
        # Convert to tensors
        fold_x_train = torch.FloatTensor(fold_x_train)
        fold_y_train = torch.FloatTensor(fold_y_train)
        fold_x_val = torch.FloatTensor(fold_x_val)
        fold_y_val = torch.FloatTensor(fold_y_val)
        
        density = torch.FloatTensor(density).cuda()
        bins = torch.FloatTensor(bins).cuda()
        
        # Setting up the dataset
        print("Setting up dataset")
        fold_x_val, _ = scaler_all_channel(fold_x_train, fold_x_val)
        scaled_test_x, _ = scaler_all_channel(fold_x_train, new_x_test)
        fold_x_train, scaler = scaler_all_channel(fold_x_train, fold_x_train)
        
        fold_y_train = torch.FloatTensor(fold_y_train)
        fold_y_val = torch.FloatTensor(fold_y_val)
    
        # Model initialization
        model = model_class()
        model = nn.DataParallel(model)
        model.cuda()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        # Create DataLoaders
        dataset = torch.utils.data.TensorDataset(torch.tensor(fold_x_train), fold_y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        dataset = torch.utils.data.TensorDataset(torch.tensor(fold_x_val), fold_y_val)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # TRAINING
        print("Start training")
        pbar = tqdm(range(total_epoches), leave=False)
        
        for epoch in pbar:
            if early_stop:
                continue
                
            model.train()
            for i, data in enumerate(train_loader):
                x_input, apneic_groundtruth = data
                x_input, apneic_groundtruth = x_input.cuda(), apneic_groundtruth.cuda()
                x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()
                apneic_stage = model(x_input)
                apneic_groundtruth = apneic_groundtruth.float()
                
                apneic_loss = nn.MSELoss()(apneic_stage, apneic_groundtruth)
                
#                 weight = density[torch.searchsorted(bins, apneic_groundtruth)] #
#                 apneic_loss = torch.sum(weight * (apneic_stage - apneic_groundtruth) ** 2) #

                optim.zero_grad()
                apneic_loss.backward()
                optim.step()
                epoch_loss.append(apneic_loss.item())            

            model.eval()
            for i, val_data in enumerate(val_loader):
                val_input, val_apneic = val_data
                val_input, val_apneic =  val_input.cuda(), val_apneic.cuda()
                val_input, val_apneic = val_input.float(), val_apneic.long()
                est_val_apneic = model(val_input)
                val_apneic = val_apneic.float()
                
                v_apneic_loss = nn.MSELoss()(est_val_apneic, val_apneic)
                
#                 weight = density[torch.searchsorted(bins, val_apneic)] #
#                 v_apneic_loss = torch.sum((1 / weight) * (est_val_apneic - val_apneic) ** 2) #

                val_epoch_loss.append(v_apneic_loss.item())

                val_pred = est_val_apneic.float().detach()
                val_mae = mean_absolute_error(val_apneic.cpu(), val_pred.cpu())
                val_mse = mean_squared_error(val_apneic.cpu(), val_pred.cpu())
                val_r2 = r2_score(val_apneic.cpu(), val_pred.cpu())
                val_epoch_mae.append(val_mae)
                val_epoch_mse.append(val_mse)
                val_epoch_r2.append(val_r2)

            t_loss = np.mean(epoch_loss)
            v_loss = np.mean(val_epoch_loss)
            fold_train_loss.append(t_loss)
            fold_val_loss.append(v_loss)
            
            v_mae = np.mean(val_epoch_mae)
            v_mse = np.mean(val_epoch_mse)
            v_r2 = np.mean(val_epoch_r2)
            fold_val_mae.append(v_mae)
            fold_val_mse.append(v_mse)
            fold_val_r2.append(v_r2)
            
            if round(v_mse, 3) < round(min_val_mse, 3):
                torch.save(model, f"{weight_dir}/recentmodel_test_leaveSubject_{model_name}.pt")
                epochs_no_improve = 0
                min_val_mse = v_mse
            else:
                epochs_no_improve += 1
                
            pbar.set_description("Loss %.3f, Val_loss %.3f, Val_mae %.3f, Val_mse %.3f, Val_r2 %.5f, No improve %d" 
             % (t_loss, v_loss, v_mae, v_mse, v_r2, epochs_no_improve))

            if epochs_no_improve == n_epochs_stop:
                print(f'Early stopping at epoch {epoch}!')
                early_stop = True
                last_epoch = epoch

        # PLOT LOSS
        fig = plt.figure()
        plt.plot(fold_train_loss, label="Training Loss")
        plt.plot(fold_val_loss, label="Validation Loss")
        plt.legend()
        plot_save_dir = log_dir + f'/learn_curve_{dataset_name}_{model_name}_{outer_fold}-{fold}.png'
        plt.savefig(plot_save_dir)
        print(f"Saved learning loss plot to {plot_save_dir}")

        # TESTING
        model = torch.load(f"{weight_dir}/recentmodel_test_leaveSubject_{model_name}.pt")
        scaled_float_test_x = torch.tensor(scaled_test_x).float()
        new_y_test = new_y_test.float()

        start = time.process_time()
        test_pred = model(scaled_float_test_x)
        inference_time = time.process_time() - start
        test_pred = test_pred.float().detach().cpu().numpy()
#         test_pred = y_transformer.inverse_transform(test_pred.reshape(-1, 1)).reshape(-1) #
        
        test_mae, test_mse, test_r2 = metrics(new_y_test.numpy(), test_pred)

        print(f'[{outer_fold}/{fold}] Test MAE: {test_mae}')

        fold_test_mae.append(test_mae)
        fold_test_mse.append(test_mse)
        fold_test_r2.append(test_r2)

        if best_mae > test_mae:
            print(f'Found new best test MAE = {test_mae}')
            model_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pt"
            torch.save(model, model_dir)
            print(f'Best model saved to {model_dir}')
            
            scaler_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}_scaler.pickle"
            with open(scaler_dir, 'wb') as handle:
                pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Corresponding scaler saved to {scaler_dir}')
            
#             scaler_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}_scaler_y.pickle"
#             with open(scaler_dir, 'wb') as handle:
#                 pickle.dump(y_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f'Corresponding y-scaler saved to {scaler_dir}')
            
            best_mae = test_mae
            best_mse = test_mse
            best_r2 = test_r2
            best_inference_time = inference_time
            best_test_pred = test_pred
            
            if save_pred:
                arranged_pred = arrange_save_pred(test_set[1], test_pred)
                np.savez(log_dir + f'/pred_{dataset_name}_{model_name}_{outer_fold}', pred=arranged_pred, infer_time=inference_time)
        
        print('=' * 30)

    mae_msg = gen_result(fold_test_mae, "MAE", factor=1)
    mse_msg = gen_result(fold_test_mse, "MSE", factor=1)
    r2_msg = gen_result(fold_test_r2, "R2", factor=1)
    inf_msg = f"Inference time: {best_inference_time:.5f} for {len(new_y_test)} samples"
    result = '\n'.join([mae_msg, mse_msg, r2_msg, inf_msg])
    print(result)
    
    with open(log_dir + f'/best_{dataset_name}_{model_name}_{outer_fold}.txt', 'w') as f:
        f.write(result)

    
def gen_result(result, name, factor=100):
    text = f"{name}: {np.mean(result)*factor:.3f} / {np.min(result)*factor:.3f} / {np.max(result)*factor:.3f} "
    text += f"AVG:SD = {np.average(result)*factor:.3f} ± {np.std(result)*factor:.3f}"
    return text


def forward_ml(
    test_set,
    model_class, model_name,
    outer_fold, dataset_name,
    log_dir, weight_dir,
    win_size=60, wavenet_ch=7
):
    test_x, _ = test_set
    new_x_test = np.vstack(test_x)
    
    estimator = pickle.load(open(f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pickle", 'rb'))
    
    start = time.process_time()
    test_pred = estimator.predict(new_x_test)
    inference_time = time.process_time() - start
    
    return test_pred, inference_time
    
    
def forward(
    test_set, 
    model_class, model_name, 
    outer_fold, dataset_name,
    log_dir, weight_dir,
    device,
    train_set=None,
    win_size=60, wavenet_ch=7
):
    test_x, _ = test_set
    
    new_x_test = torch.FloatTensor(np.vstack(test_x))
    
    # Model initialization
    model = model_class()
    model = nn.DataParallel(model)
    model.cuda()
    
    scalers = pickle.load(open(f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}_scaler.pickle", 'rb'))
    model = torch.load(f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pt")
    model.eval()
    
    # Setting up the dataset
    print("Setting up dataset")
    scaled_x_test, _ = scaler_all_channel(test_x=new_x_test, scalers=scalers)
    scaled_x_test = torch.tensor(scaled_x_test).float()

    start = time.process_time()
    test_pred = model(scaled_x_test)
    inference_time = time.process_time() - start

    test_pred = test_pred.float()
    
    return test_pred.cpu().numpy(), inference_time
    

def evaluate(test_y, test_pred, args, outer_fold):
    test_y = np.vstack(test_y)
    test_y = make_y(test_y)
        
    test_mae, test_mse, test_r2 = metrics(test_y, test_pred)
    
    return test_mae, test_mse, test_r2
