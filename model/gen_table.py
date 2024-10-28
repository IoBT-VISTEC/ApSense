import hydra
import mlflow

METRICS = {
    "acc": "Accuracy",
    "f1": "Macro F1",
    "sens": "Sensitivity",
    "spec": "Specificity",
    "auroc": "AUROC"
}
MODELS = {
    "EEGNetSA": "PPGNetSA",
    "SingleAIOSA": "AIOSA",
    "RRWaveNet": "RRWaveNet",
    "DSepST15Net": "ApSense",
    "DSepST15Net_no_branch": "ApSense",
}
MODEL_ORDER = ["RRWaveNet", "PPGNetSA", "AIOSA", "ApSense"]
COLUMNS = {
    "tags.dataset": "dataset",
    "tags.model": "model",
}
DATASETS = {
    "mesa": "MESA",
    "heartbeat": "HeartBEAT",
}


@hydra.main(version_base="1.2", config_path="config/", config_name="evaluate")
def main(config):
    mlflow.set_tracking_uri(config.tracking_uri)

    experiment_id = mlflow.set_experiment(
        experiment_name=config.exp_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=f"{experiment_id}")
    runs = runs.rename(columns=COLUMNS)

    runs["model"] = runs["model"].replace(MODELS).astype("category")
    runs["model"] = runs["model"].cat.set_categories(MODEL_ORDER)
    runs = runs.sort_values("model")

    for metric, name in METRICS.items():
        metric = f"metrics.{metric}"
        runs[name] = (runs[f"{metric}_mean"] * 100).round(2).apply("{:.2f}".format) + " $\pm$ " + \
            (runs[f"{metric}_std"] * 100).apply("{:.2f}".format)

    for i, (dataset, rows) in enumerate(runs.groupby("dataset"), 1):
        print("\multirow{4}{*}{\emph{" + DATASETS[dataset] + "}}")
        rows = rows.to_records()

        for row in rows:
            line = "& " + \
                " & ".join([row["model"]] + [row[name]
                           for name in METRICS.values()]) + " \\\\"
            print(line)

        if i < len(DATASETS):
            print("\midrule[0.1em]")


if __name__ == "__main__":
    main()
