import hydra
import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

from runner import load_dataset, evaluate_onset, evaluate_severity, forward, bland_altman_plot, get_ahis
from models import MODELS
from utils import config_gpu

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config/", config_name="evaluate")
def main(config):
    log.info("Initializing...")
    mlflow.set_tracking_uri(config.tracking_uri)

    ### GPU Configuration ###
    device = config_gpu(config)

    # Start Running
    log.info("Starting the evaluation...")
    results = []

    all_gt_ahis = []
    all_pred_ahis = []

    experiment_id = mlflow.set_experiment(
        experiment_name=config.exp_name).experiment_id

    log.info("Starting mlflow run...")
    with mlflow.start_run(experiment_id=experiment_id, run_name=config.model):
        for FOLD in range(int(config.num_folds)):
            train_x, train_y, test_x, test_y = load_dataset(
                config.dataset, fold=FOLD, data_dir=config.dataset_dir, features=config.features)

            if config.save_pred:
                result = np.load(
                    f"{config.save_pred}/{FOLD}", allow_pickle=True)
                infer_time = result["infer_time"]
                test_pred = result["pred"]
            else:
                test_pred, infer_time, test_pred_prob = forward(
                    train_set=(train_x, train_y),
                    test_set=(test_x, test_y),
                    model_class=MODELS[config.model],
                    model_name=config.model,                 # for naming purpose
                    dataset_name=config.dataset,             # for naming purpose
                    outer_fold=FOLD,
                    log_dir=config.log_dir,                  # save your plots
                    weight_dir=config.weight_dir,            # save your plots
                    device=device,
                    wavenet_ch=train_x[0].shape[1]
                )

            if config.mode == "onset":
                result = evaluate_onset(
                    test_y, test_pred, test_pred_prob, config, FOLD)

            elif config.mode == "severity":
                result = evaluate_severity(
                    test_y, test_pred, config, FOLD)

            result["infer_time"] = infer_time / sum([len(a) for a in test_y])
            results.append(result)

            gt_ahis, pred_ahis = get_ahis(test_y, test_pred, config, FOLD)
            all_gt_ahis.append(gt_ahis)
            all_pred_ahis.append(pred_ahis)

        # Concatenate AHIs
        all_gt_ahis = np.concatenate(all_gt_ahis, axis=None)
        all_pred_ahis = np.concatenate(all_pred_ahis, axis=None)

        mlflow.set_tags(config)

        # Scatter
        fig = plt.figure(figsize=(6, 6), dpi=180)
        plt.scatter(all_gt_ahis, all_pred_ahis, s=3, alpha=0.4)
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        mlflow.log_figure(fig, "scatter_all.png")

        # Bland-Altman
        fig = bland_altman_plot(all_gt_ahis, all_pred_ahis, config)
        mlflow.log_figure(fig, "bland_altman_all.png")

        pearson = pearsonr(all_gt_ahis, all_pred_ahis)
        mlflow.log_metric("pearson_corr", pearson[0])
        mlflow.log_metric("pearson_pvalue", pearson[1])

        results = pd.DataFrame.from_records(results)

        # Mean
        summary = results.mean(axis=0).to_dict()
        mlflow.log_metrics({f"{k}_mean": v for k, v in summary.items()})

        # Std
        summary = results.std(axis=0, ddof=0).to_dict()
        mlflow.log_metrics({f"{k}_std": v for k, v in summary.items()})


if __name__ == "__main__":
    main()
