import hydra
import logging
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGB

from runner import load_dataset, run
from models import MODELS
from utils import config_gpu

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config/", config_name="train")
def main(config):
    log.info("Initializing...")

    ### GPU Configuration ###
    device = config_gpu(config)

    # TODO: Import by files
    lr = 1e-3
    batch_size = 1024

    # Start Running
    log.info("Starting the training...")
    for FOLD in range(config.starting_fold, config.starting_fold + config.num_folds):
        all_x, all_y, test_x, test_y = load_dataset(
            config.dataset, fold=FOLD, data_dir=config.dataset_dir, features=config.features)
        wavenet_ch = all_x[0].shape[1]

        run(
            train_set=(all_x, all_y),
            test_set=(test_x, test_y),
            # refer to "from models import *"
            model_class=MODELS[config.model],
            model_name=config.model,                 # for naming purpose
            dataset_name=config.dataset,             # for naming purpose
            outer_fold=FOLD,
            log_dir=config.log_dir,                  # save your plots
            weight_dir=config.weight_dir,            # save your plots
            device=device,
            subsampling=config.subsampling,

            lr=lr,
            batch_size=batch_size,
            wavenet_ch=wavenet_ch,
        )


if __name__ == "__main__":
    main()
