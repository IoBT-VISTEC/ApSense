import logging

import os
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # ADD

import pickle

from tqdm import tqdm
import time

import torch
from torch import nn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, f1_score, confusion_matrix, r2_score, roc_auc_score
)
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import RandomUnderSampler
from losses import *

import shap  #
shap.initjs()


# for consistency, all seeds are set to 69420
seed = 69420
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

log = logging.getLogger(__name__)


def ends(seq):
    curr = 0
    for item in seq:
        if item != 0:
            curr = 1
        else:  # reverts from 1 to 0
            if curr == 1:
                return 1
    return 0


def make_classif_y(all_y, conditioned=True):
    classif_y = []
    for window in all_y:
        w = window[:len(window) // 2]  # first half
        if w.sum() > 0 and ends(w) == 1:  # positive and ends
            classif_y.append(1)
        else:
            classif_y.append(0)
    return np.array(classif_y)


def subsample(X, y):
    rus = RandomUnderSampler(random_state=42)
    x_shape = X.shape
    X_res = X.reshape(x_shape[0], -1)
    X_res, y_res = rus.fit_resample(X_res, y)
    X_res = X_res.reshape(-1, *(x_shape[1:]))
    log.info(f"Remaining samples: {np.unique(y_res, return_counts=True)}")
    return X_res, y_res


def scaler_all_channel(train_x=None, test_x=None, scalers=None, ch=-1):
    prescaled = scalers is not None

    if (train_x is None and not prescaled) or (train_x is not None and prescaled):
        raise ValueError("Only train_x or scalers is required.")

    if test_x is None:
        test_x = train_x

    scalers = scalers if scalers else []
    scaled_X = []
    num_ch = train_x.shape[1] if train_x is not None else test_x.shape[1]

    if ch == -1:  # SCALER ALL CHANNEL
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


def load_dataset(dataset, fold, data_dir='/mount/guntitats/apsens_processed', features=None):
    def filter_features(data_list, feature_idx):
        return [data[:, feature_idx, :] for data in data_list]

    try:
        x_train_file = data_dir + f'/{dataset}_fold{fold}_x_train.pickle'
        y_train_file = data_dir + f'/{dataset}_fold{fold}_y_train.pickle'
        x_test_file = data_dir + f'/{dataset}_fold{fold}_x_test.pickle'
        y_test_file = data_dir + f'/{dataset}_fold{fold}_y_test.pickle'

        all_x = pd.read_pickle(x_train_file)
        all_y = pd.read_pickle(y_train_file)
        test_x = pd.read_pickle(x_test_file)
        test_y = pd.read_pickle(y_test_file)

        feature_map = {
            "pwa": 0,
            "spd": 1,
            "dpd": 2,
            "pa": 3,
            "ppi": 4,
            "dpwa": 5,
            "dppi": 6
        }

        if features is not None and len(features):
            features = [feature_map[feature] for feature in features]
            all_x = filter_features(all_x, features)
            test_x = filter_features(test_x, features)

        return all_x, all_y, test_x, test_y

    except:
        raise FileNotFoundError(
            f"Some files do not exist, have incorrect format or naming. Please check README.")


class stdard(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalers = []

    def fit(self, X, y=None):
        if len(X.shape) == 2:
            X = np.expand_dims(X, 1)

        _, self.scalers = scaler_all_channel(X)
        return self

    def transform(self, X, y=None):
        if len(X.shape) == 2:
            X = np.expand_dims(X, 1)

        scaled_X = []

        for j in range(X.shape[1]):
            scaled_X.append(self.scalers[j].transform(X[:, j, :]))

        scaled_X = np.array(scaled_X)
        scaled_X = np.swapaxes(scaled_X, 0, 1)
        concat_X = scaled_X.reshape(-1, scaled_X.shape[1] * scaled_X.shape[2])

        return concat_X


def get_emb_prediction(model, loader, device, l2Norm=True):
    with torch.no_grad():
        model.eval()
        pred_embs = []
        labels = []

        for i, data in enumerate(loader):
            x_input, apneic_groundtruth = data
            x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()
            apneic_stage = model(x_input.to(
                device), l2Norm=l2Norm).cpu().numpy()
            pred_embs.append(apneic_stage)
            labels.append(apneic_groundtruth.numpy())

        return np.vstack(pred_embs), np.hstack(labels)


def get_prediction_from_embedding(tr_emb, te_emb, tr_lbl, save_path=None, thresh=1):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(tr_emb)
    distances, indices = nbrs.kneighbors(te_emb)

    pred = [int(np.sum(tr_lbl[index]) > thresh) for index in indices]
    pred_raw = np.array([list(tr_lbl[index]) for index in indices])
    if save_path is None:
        pass
    else:
        np.save(save_path, pred_raw)
    return torch.tensor(pred)


def metrics(gt, pd):
    acc = accuracy_score(gt, pd)
#     f1 = f1_score(gt, pd, average='weighted')
    f1 = f1_score(gt, pd, average='macro')
    sen = np.sum(gt * pd) / np.sum(gt)
    spec = (len(gt) - np.count_nonzero(gt + pd)) / (len(gt) - np.sum(gt))
    cm = confusion_matrix(gt, pd)
    return acc, f1, sen, spec, cm


def save_cm(con_mat, dataset_name, model_name, outer_fold, log_dir, fold=None, severity=False):
    fig = plt.figure()
    sns.heatmap(con_mat/np.sum(con_mat), annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if fold:
        plt.title("Test Set")
        cm_save_dir = f'{dataset_name}_{model_name}_{outer_fold}-{fold}.png'
    elif severity:
        plt.title("Severity Classification")
        cm_save_dir = f'{dataset_name}_{model_name}_{outer_fold}_best_severity.png'
    else:
        plt.title("Best Test Combination")
        cm_save_dir = f'{dataset_name}_{model_name}_{outer_fold}_best.png'

    mlflow.log_figure(fig, f"cm/{cm_save_dir}")
    log.info(f"Saved confusion matrix to {cm_save_dir}")


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
    log.info("Setting up dataset")
    all_x, all_y = train_set
    test_x, test_y = test_set

    all_x = np.array([inx for outx in all_x for inx in outx])
    all_y = np.array([iny for outy in all_y for iny in outy])
    new_x_test = np.vstack(test_x)
    new_y_test = np.vstack(test_y)

    all_y = make_classif_y(all_y)
    new_y_test = make_classif_y(new_y_test)

    if subsampling:
        all_x, all_y = subsample(all_x, all_y)

    splits = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    args = {'class_weight': 'balanced'}

    # Prevents too-long runs
    if model_name == 'SVC':
        args['max_iter'] = 10000

    if model_name == 'XGB':
        args = {'scale_pos_weight': np.sum(
            all_y) / (len(all_y) - np.sum(all_y))}

    estimators = [('std', stdard()), ('reduce_dim', PCA(
        n_components=0.95)), ('clf', model_class(**args))]
    train_pipe = Pipeline(estimators)
    clf = GridSearchCV(train_pipe, grid_params, cv=splits,
                       scoring='f1_micro', n_jobs=-1, verbose=10)

    clf.fit(all_x, all_y)
    log.info(f'Best score from grid search: {clf.best_score_}')

    start = time.process_time()
    test_pred = clf.best_estimator_.predict(new_x_test)
    inference_time = time.process_time() - start

    test_acc, test_f1, test_sensitivity, test_specificity, con_mat = metrics(
        new_y_test, test_pred)

    model_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pickle"
    with open(model_dir, 'wb') as handle:
        pickle.dump(clf.best_estimator_, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f'Best model saved to {model_dir}')

    acc_msg = f"Accuracy: {test_acc}"
    f1_msg = f"F1: {test_f1}"
    sens_msg = f"Sensitivity: {test_sensitivity}"
    spec_msg = f"Specificity: {test_specificity}"
    inf_msg = f"Inference time: {inference_time:.5f} for {len(new_y_test)} samples"
    result = '\n'.join([acc_msg, f1_msg, sens_msg, spec_msg, inf_msg])
    log.info(result)

    with open(log_dir + f'/best_{dataset_name}_{model_name}_{outer_fold}.txt', 'w') as f:
        f.write(result)

    save_cm(con_mat, dataset_name, model_name, outer_fold, log_dir)

    if save_pred:
        arranged_pred = arrange_save_pred(test_set[1], test_pred)
        np.savez(log_dir + f'/pred_{dataset_name}_{model_name}_{outer_fold}.npy',
                 pred=arranged_pred, infer_time=inference_time)


def run(
    train_set, test_set,
    model_class, model_name,
    outer_fold, dataset_name,
    log_dir, weight_dir,
    device,
    subsampling=False,
    save_pred=False,
    lr=1e-3, batch_size=1024, total_epoches=10000, n_epochs_stop=30, win_size=60, wavenet_ch=7, save_emb_pred=True
):
    all_x, all_y = train_set
    test_x, test_y = test_set

    splits = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    best_test_acc = []
    best_test_f1 = []
    best_acc = 0
    best_f1 = 0

    fold_test_acc = []
    fold_test_f1 = []
    fold_test_spec = []
    fold_test_sens = []
    thres = 0.0

    all_x = np.array([inx for outx in all_x for inx in outx])
    all_y = np.array([iny for outy in all_y for iny in outy])
    all_y = make_classif_y(all_y)

    if subsampling:
        all_x, all_y = subsample(all_x, all_y)

    new_x_test = torch.FloatTensor(np.vstack(test_x))
    new_y_test = torch.FloatTensor(np.vstack(test_y))
    new_y_test = torch.FloatTensor(make_classif_y(new_y_test))

    for fold, (train_ids, val_ids) in enumerate(splits.split(all_x, all_y)):
        log.info(f'Running fold {outer_fold}-{fold}')
        log.info(
            f'Train {len(train_ids)} samples | Validation {len(val_ids)} samples')

        fold_x_train = all_x[train_ids]
        fold_y_train = all_y[train_ids]
        fold_x_val = all_x[val_ids]
        fold_y_val = all_y[val_ids]

        # Trackers
        fold_train_loss, fold_val_loss, fold_val_acc, fold_val_f1 = [], [], [], []
        epoch_loss, val_epoch_loss, val_epoch_acc, val_epoch_prec, val_epoch_f1 = [], [], [], [], []
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = 10000
        max_val_f1 = 0

        # Convert to tensors
        fold_x_train = torch.FloatTensor(fold_x_train)
        fold_y_train = torch.FloatTensor(fold_y_train)
        fold_x_val = torch.FloatTensor(fold_x_val)
        fold_y_val = torch.FloatTensor(fold_y_val)

        # Setting up the dataset
        log.info("Setting up dataset")
        fold_x_val, _ = scaler_all_channel(fold_x_train, fold_x_val)
        scaled_test_x, _ = scaler_all_channel(fold_x_train, new_x_test)
        fold_x_train, scaler = scaler_all_channel(fold_x_train, fold_x_train)

        fold_y_train = torch.FloatTensor(fold_y_train)
        fold_y_val = torch.FloatTensor(fold_y_val)

        # Model initialization
        model = model_class(num_channels=wavenet_ch, winsize=win_size)
        model = nn.DataParallel(model)
        model.cuda()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        class_weights = compute_class_weight(
            'balanced', np.unique(fold_y_train), np.array(fold_y_train))
        log.info(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        class_weights = class_weights.cuda()

        # Create DataLoaders
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(fold_x_train), fold_y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(fold_x_val), fold_y_val)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        # TRAINING
        log.info("Start training")
        pbar = tqdm(range(total_epoches), leave=False)

        DSEPEMB_FLAG = model_name == 'DSepEmbedder'

        if DSEPEMB_FLAG:
            CustomLoss = TripletLoss(device)

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
                data_weight = torch.where(
                    apneic_groundtruth == 0, class_weights[0], class_weights[1])

                if DSEPEMB_FLAG:
                    apneic_loss = CustomLoss(apneic_stage, apneic_groundtruth)
                else:
                    apneic_loss = nn.BCEWithLogitsLoss(weight=data_weight)(
                        apneic_stage, apneic_groundtruth)

                optim.zero_grad()
                apneic_loss.backward()
                optim.step()
                epoch_loss.append(apneic_loss.item())

            model.eval()
            for i, val_data in enumerate(val_loader):
                val_input, val_apneic = val_data
                val_input, val_apneic = val_input.cuda(), val_apneic.cuda()
                val_input, val_apneic = val_input.float(), val_apneic.long()
                est_val_apneic = model(val_input)
                val_apneic = val_apneic.float()
                val_weight = torch.where(
                    val_apneic == 0, class_weights[0], class_weights[1])
                if not DSEPEMB_FLAG:
                    v_apneic_loss = nn.BCEWithLogitsLoss(
                        weight=val_weight)(est_val_apneic, val_apneic)
                else:
                    v_apneic_loss = CustomLoss(est_val_apneic, val_apneic)
                val_epoch_loss.append(v_apneic_loss.item())

                # Val Accuracy
                if not DSEPEMB_FLAG:
                    threshold = torch.tensor([thres]).cuda().to(device)
                    val_pred = (est_val_apneic > threshold).float()*1
                    val_acc = accuracy_score(val_apneic.cpu(), val_pred.cpu())
                    val_prec = precision_score(
                        val_apneic.cpu(), val_pred.cpu())
                    val_f1 = f1_score(val_apneic.cpu(), val_pred.cpu())
                    val_epoch_prec.append(val_prec)
                    val_epoch_acc.append(val_acc)
                    val_epoch_f1.append(val_f1)

            t_loss = np.mean(epoch_loss)
            v_loss = np.mean(val_epoch_loss)
            fold_train_loss.append(t_loss)
            fold_val_loss.append(v_loss)

            if not DSEPEMB_FLAG:
                v_acc = np.mean(val_epoch_acc)
                v_prec = np.mean(val_epoch_prec)
                v_f1 = np.mean(val_epoch_f1)
                fold_val_acc.append(v_acc)
                fold_val_f1.append(v_f1)

#             if DSEPEMB_FLAG:
            if round(v_loss, 3) < round(min_val_loss, 3):
                torch.save(
                    model, f"{weight_dir}/recentmodel_test_leaveSubject_{model_name}.pt")
                epochs_no_improve = 0
                min_val_loss = v_loss
            else:
                epochs_no_improve += 1

#             else:
#                 if round(v_f1, 3) > round(max_val_f1, 3):
#                     torch.save(model, f"{weight_dir}/recentmodel_test_leaveSubject_{model_name}.pt")
#                     epochs_no_improve = 0
#                     max_val_f1 = v_f1
#                 else:
#                     epochs_no_improve += 1

            if DSEPEMB_FLAG:
                pbar.set_description("Loss %.3f, Val_loss %.3f, No improve %d" % (
                    t_loss, v_loss, epochs_no_improve))
            else:
                pbar.set_description("Loss %.3f, Val_loss %.3f, Val_acc %.3f, Val_prec %.3f, val_f1 %.5f, No improve %d"
                                     % (t_loss, v_loss, v_acc, v_prec, v_f1, epochs_no_improve))

            if epochs_no_improve == n_epochs_stop:
                log.info(f'Early stopping at epoch {epoch}!')
                early_stop = True
                last_epoch = epoch

        # PLOT LOSS
        fig = plt.figure()
        plt.plot(fold_train_loss, label="Training Loss")
        plt.plot(fold_val_loss, label="Validation Loss")
        plt.legend()
        plot_save_dir = log_dir + \
            f'/learn_curve_{dataset_name}_{model_name}_{outer_fold}-{fold}.png'
        plt.savefig(plot_save_dir)
        log.info(f"Saved learning loss plot to {plot_save_dir}")

        # add december 11-12-66
        np.savez(log_dir + f'/collected_loss_{dataset_name}_{model_name}_{outer_fold}-{fold}.npz', fold_train_loss=fold_train_loss,
                 fold_val_loss=fold_val_loss)

        # Plot gradient flow 27-11-66
#         fig = plt.figure()
        # plot_grad_flow_v2(model.named_parameters(), log_dir, dataset_name, model_name, outer_fold, fold)
#         plot_save_dir = log_dir + f'/grad_flow_{dataset_name}_{model_name}_{outer_fold}-{fold}.png'
#         plt.savefig(plot_save_dir)
#         log.info(f"Saved grad flow plot to {plot_save_dir}")

        # TESTING
        del model
        del train_loader, val_loader

        model = torch.load(
            f"{weight_dir}/recentmodel_test_leaveSubject_{model_name}.pt")
        scaled_float_test_x = torch.tensor(scaled_test_x).float()
        new_y_test = new_y_test.float()

        # Get inference time
        if DSEPEMB_FLAG:
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(scaled_float_test_x), new_y_test.float())
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True)

            train_emb, train_y = get_emb_prediction(
                model, train_loader, device)
            start = time.process_time()
            test_emb, test_y = get_emb_prediction(model, test_loader, device)

            if not save_emb_pred:
                emb_path = None
            else:
                try:
                    os.mkdir(f'{log_dir}/pred_{model_name}')
                except:
                    pass
                emb_path = f'{log_dir}/pred_{model_name}/pred_{dataset_name}_{model_name}_{outer_fold}-{fold}.npy'

            test_pred = get_prediction_from_embedding(
                train_emb, test_emb, train_y, save_path=emb_path)
            inference_time = time.process_time() - start
        else:
            dataset = torch.utils.data.TensorDataset(
                scaled_float_test_x, new_y_test)
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False)
            test_pred = []

            start = time.process_time()

            model.eval()
            with torch.no_grad():
                for i, test_batch in enumerate(test_loader):
                    x_test_batch, y_test_batch = test_batch
                    x_test_batch = x_test_batch.cuda().float()
                    test_pred.append(model(x_test_batch))

            inference_time = time.process_time() - start
            test_pred = torch.cat(test_pred)
            threshold = torch.tensor([thres]).cuda().to(device)
            test_pred = (test_pred > threshold).float()*1

        test_acc, test_f1, test_sensitivity, test_specificity, con_mat = metrics(
            new_y_test.numpy(), test_pred.cpu().numpy())

        log.info(f'[{outer_fold}/{fold}] Test Accuracy: {test_acc}')
        log.info(classification_report(new_y_test, test_pred.cpu()))

        save_cm(con_mat, dataset_name, model_name,
                outer_fold, log_dir, fold=fold)

        fold_test_acc.append(test_acc)
        fold_test_f1.append(test_f1)
        fold_test_spec.append(test_specificity)
        fold_test_sens.append(test_sensitivity)

        if best_f1 < test_f1:
            log.info(f'Found new best test f1 = {test_f1}')
            model_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pt"
            torch.save(model, model_dir)
            log.info(f'Best model saved to {model_dir}')

            scaler_dir = f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}_scaler.pickle"
            with open(scaler_dir, 'wb') as handle:
                pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f'Corresponding scaler saved to {scaler_dir}')

            best_acc = test_acc
            best_f1 = test_f1
            best_inference_time = inference_time
            best_test_pred = test_pred
            best_con_mat = confusion_matrix(new_y_test, test_pred.cpu())

            if save_pred:
                arranged_pred = arrange_save_pred(
                    test_set[1], test_pred.cpu().numpy())
                np.savez(log_dir + f'/pred_{dataset_name}_{model_name}_{outer_fold}.npy',
                         pred=arranged_pred, infer_time=inference_time)

        log.info('=' * 30)

    acc_msg = gen_result(fold_test_acc, "Accuracy")
    f1_msg = gen_result(fold_test_f1, "F1")
    sens_msg = gen_result(fold_test_sens, "Sensitivity")
    spec_msg = gen_result(fold_test_spec, "Specificity")
    inf_msg = f"Inference time: {best_inference_time:.5f} for {len(new_y_test)} samples"
    result = '\n'.join([acc_msg, f1_msg, sens_msg, spec_msg, inf_msg])
    log.info(result)

    with open(log_dir + f'/best_{dataset_name}_{model_name}_{outer_fold}.txt', 'w') as f:
        f.write(result)

    save_cm(best_con_mat, dataset_name, model_name, outer_fold, log_dir)


def gen_result(result, name, factor=100):
    text = f"{name}: {np.mean(result)*factor:.2f} / {np.min(result)*factor:.2f} / {np.max(result)*factor:.2f} "
    text += f"AVG:SD = {np.average(result)*factor:.2f} ± {np.std(result)*factor:.2f}"
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

    estimator = pickle.load(
        open(f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pickle", 'rb'))

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
    thres = 0.0

    new_x_test = torch.FloatTensor(np.vstack(test_x))

    # Model initialization
    model = model_class(num_channels=wavenet_ch, winsize=win_size)
    model = nn.DataParallel(model)
    model.cuda()

    scalers = pickle.load(open(
        f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}_scaler.pickle", 'rb'))
    model = torch.load(
        f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pt")
    model.eval()

    # Setting up the dataset
    log.info("Setting up dataset")
    scaled_x_test, _ = scaler_all_channel(test_x=new_x_test, scalers=scalers)
    scaled_x_test = torch.tensor(scaled_x_test).float()

    # Get predictions
    if model_name == "DSepEmbedder":
        train_x, train_y = train_set

        train_x = np.vstack(train_x)
        scaled_x_train, _ = scaler_all_channel(test_x=train_x, scalers=scalers)
        scaled_x_train = torch.tensor(scaled_x_test).float()

        train_y = torch.FloatTensor(make_classif_y(np.vstack(train_y)))

        test_y = test_set[1]
        test_y = torch.FloatTensor(make_classif_y(np.vstack(test_y)))

        log.info(scaled_x_train.shape, train_y.shape)
        dataset = torch.utils.data.TensorDataset(scaled_x_train, train_y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        dataset = torch.utils.data.TensorDataset(scaled_x_test, test_y)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)

        train_emb, train_y = model.get_emb_prediction(train_loader, device)
        start = time.process_time()
        test_emb, test_y = model.get_emb_prediction(test_loader, device)
        test_pred = get_prediction_from_embedding(train_emb, test_emb, train_y)
        inference_time = time.process_time() - start
    else:
        dataset = torch.utils.data.TensorDataset(scaled_x_test, scaled_x_test)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False)
        test_pred = []

        start = time.process_time()

        model.eval()
        with torch.no_grad():
            for i, test_batch in enumerate(test_loader):
                x_test_batch, _ = test_batch
                x_test_batch = x_test_batch.cuda().float()
                test_pred.append(model(x_test_batch))

        inference_time = time.process_time() - start
        test_pred = torch.cat(test_pred)

    threshold = torch.tensor([thres]).cuda().to(device)
    test_pred_prob = 1 / (1 + np.exp(-test_pred.cpu().numpy()))
    test_pred = (test_pred > threshold).float()*1

    return test_pred.cpu().numpy(), inference_time, test_pred_prob


def forward_finetune(
    test_set,
    model_class, model_name,
    outer_fold, dataset_name,
    log_dir, weight_dir,
    device,
    finetune_size=10,
    train_set=None,
    win_size=60, wavenet_ch=7
):
    test_x, test_y = test_set
    thres = 0.0

    accs = []
    f1s = []
    sens = []
    specs = []

    d = {"gt": [], "pred": [], "features": []}

    for test_x_subj, test_y_subj in tqdm(zip(test_x, test_y), total=len(test_x)):
        assert len(test_x_subj) == len(test_y_subj)

        test_y_subj = make_classif_y(test_y_subj).reshape(-1)

        # Model initialization
        model = model_class()
        model = nn.DataParallel(model)
        model.cuda()

        scalers = pickle.load(open(
            f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}_scaler.pickle", 'rb'))
        model = torch.load(
            f"{weight_dir}/{dataset_name}_{model_name}_{outer_fold}.pt")
        model.train()

        if finetune_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                test_x_subj, test_y_subj, test_size=1-finetune_size/100, random_state=42, stratify=test_y_subj)

            X_test_ = X_test.copy()

            # Setting up the dataset
            X_train, _ = scaler_all_channel(test_x=X_train, scalers=scalers)
            X_test, _ = scaler_all_channel(test_x=X_test, scalers=scalers)

            X_train = torch.tensor(X_train).float()
            X_test = torch.tensor(X_test).float()
            y_train = torch.tensor(y_train).float()

            optim = torch.optim.Adam(model.parameters(), lr=1e-3)

            class_weights = compute_class_weight(
                'balanced', np.unique(y_train), np.array(y_train))
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            class_weights = class_weights.cuda()

            # Create DataLoaders
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=True, drop_last=True)

            pbar = tqdm(range(10), leave=False)
            for epoch in pbar:
                epoch_loss = []

                for i, data in enumerate(train_loader):
                    x_input, apneic_groundtruth = data
                    x_input, apneic_groundtruth = x_input.cuda(), apneic_groundtruth.cuda()
                    x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()

                    apneic_stage = model(x_input)
                    apneic_groundtruth = apneic_groundtruth.float()
                    data_weight = torch.where(
                        apneic_groundtruth == 0, class_weights[0], class_weights[1])

                    try:
                        _ = len(apneic_stage)
                    except:
                        apneic_stage = apneic_stage.reshape(-1)

                    apneic_loss = nn.BCEWithLogitsLoss(weight=data_weight)(
                        apneic_stage, apneic_groundtruth)
                    optim.zero_grad()
                    apneic_loss.backward()
                    optim.step()
                    epoch_loss.append(apneic_loss.item())

                pbar.set_description("Loss", np.mean(epoch_loss))

        else:
            X_test, y_test = test_x_subj, test_y_subj
            X_test_ = X_test.copy()
            X_test, _ = scaler_all_channel(test_x=X_test, scalers=scalers)

            X_test = torch.tensor(X_test).float()

        model.eval()
        start = time.process_time()
        test_pred = model(X_test)
        inference_time = time.process_time() - start

        threshold = torch.tensor([thres]).cuda().to(device)
        test_pred = (test_pred > threshold).float()*1
        test_pred = test_pred.cpu().numpy()

        acc, f1, sen, spec, _ = metrics(y_test, test_pred)
        d["features"].append(X_test_)
        d["gt"].append(y_test)
        d["pred"].append(test_pred)

        accs.append(acc)
        f1s.append(f1)
        sens.append(sen)
        specs.append(spec)

    d["features"] = np.array(d["features"])
    d["gt"] = np.array(d["gt"])
    d["pred"] = np.array(d["pred"])
    return accs, f1s, sens, specs, d


def evaluate_onset(test_y, test_pred, test_pred_prob, args, outer_fold):
    test_y = np.vstack(test_y)
    test_y = make_classif_y(test_y)

    test_acc, test_f1, test_sensitivity, test_specificity, con_mat = metrics(
        test_y, test_pred)
    test_auroc = roc_auc_score(test_y, test_pred_prob)
    save_cm(con_mat, args.dataset, args.model, outer_fold, args.log_dir)

    return {
        "acc": test_acc,
        "f1": test_f1,
        "sens": test_sensitivity,
        "spec": test_specificity,
        "auroc": test_auroc
    }


def evaluate_severity(test_y, test_pred, args, outer_fold, stride=30, ahi_threshold=7):
    gt_severes = []
    pred_severes = []

    preds = arrange_save_pred(test_y, test_pred)

    for gt, pred in zip(test_y, preds):
        assert len(gt) == len(pred)

        gt = make_classif_y(gt)

        gt_ahi = np.sum(gt) / (len(gt) * stride + stride) * 3600
        pred_ahi = np.sum(pred) / (len(pred) * stride + stride) * 3600

        gt_severes.append(int(gt_ahi >= ahi_threshold))
        pred_severes.append(int(pred_ahi >= ahi_threshold))

    log.info(gt_severes)
    log.info(pred_severes)

    test_acc, test_f1, test_sensitivity, test_specificity, con_mat = metrics(
        np.array(gt_severes), np.array(pred_severes))
    save_cm(con_mat, args.dataset, args.model,
            outer_fold, args.log_dir, severity=True)

    return {
        "acc": test_acc,
        "f1": test_f1,
        "sens": test_sensitivity,
        "spec": test_specificity,
        "auroc": 0,
    }


def get_ahis(test_y, test_pred, args, outer_fold, stride=30):
    gt_ahis = []
    pred_ahis = []

    preds = arrange_save_pred(test_y, test_pred)

    for gt, pred in zip(test_y, preds):
        assert len(gt) == len(pred)

        gt = make_classif_y(gt)

        gt_ahi = np.sum(gt) / (len(gt) * stride + stride) * 3600
        pred_ahi = np.sum(pred) / (len(pred) * stride + stride) * 3600

        gt_ahis.append(gt_ahi)
        pred_ahis.append(pred_ahi)

    fig = plt.figure(figsize=(6, 6), dpi=180)
    plt.scatter(gt_ahis, pred_ahis, s=3, alpha=0.4)
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')

    mlflow.log_figure(fig, f'scatter/{args.dataset}_{args.model}_{outer_fold}_best.png')

    return gt_ahis, pred_ahis


def bland_altman_plot(data1, data2, args, **kwargs):
    data1 = MinMaxScaler().fit_transform(data1.reshape(-1, 1)).reshape(-1)
    data2 = MinMaxScaler().fit_transform(data2.reshape(-1, 1)).reshape(-1)

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    fig = plt.figure(figsize=(6, 6), dpi=180)
    plt.scatter(mean, diff, s=8, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel("Average")
    plt.ylabel("Difference")
    return fig


def find_ls_r2(y1, y2):
    A = np.vstack([y1, np.ones(len(y1))]).T

    # Use numpy's least squares function
    m, c = np.linalg.lstsq(A, y2)[0]

    # Define the values of our least squares fit
    f = m * y1 + c

    for x, y in zip(y1, f):
        log.info(x, y)
    log.info(len(y1))

    return r2_score(y1, f)


# Add plot the gradients flow  27-11-66
def plot_grad_flow_v2(named_parameters, log_dir, dataset_name, model_name, outer_fold, fold):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    Ref:  https://github.com/alwynmathew/gradflow-check/blob/master/gradflow_check.py

    '''

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:

        #         if(p.requires_grad) and ("bias" not in n):
        #             layers.append(n)
        #             ave_grads.append(p.grad.abs().mean())
        #             max_grads.append(p.grad.abs().max())

        # New
        if p.grad is not None:
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            layers.append(n)

    plt.figure(figsize=(20, 10), dpi=180)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers,
               rotation="vertical", fontsize=10)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.tight_layout()
    plot_save_dir = log_dir + \
        f'/grad_flow_{dataset_name}_{model_name}_{outer_fold}-{fold}.png'
    plt.savefig(plot_save_dir)

#     np.save(log_dir + f'ave_grads_{dataset_name}_{model_name}_{outer_fold}-{fold}.npy', np.array(ave_grads), allow_pickle=True)
#     np.save(log_dir + f'max_grads_{dataset_name}_{model_name}_{outer_fold}-{fold}.npy', np.array(max_grads), allow_pickle=True)
#     np.save(log_dir + f'layers_{dataset_name}_{model_name}_{outer_fold}-{fold}.npy', np.array(layers), allow_pickle=True)

    np.savez(log_dir + f'/GradientFlow_{dataset_name}_{model_name}_{outer_fold}-{fold}.npz', ave_grads=np.array(ave_grads),
             max_grads=np.array(max_grads), layers=np.array(layers))
