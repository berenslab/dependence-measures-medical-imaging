import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import wandb
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import Subset

from src.models.trainer import get_trainer
from src.utils.test import test_model
from src.utils.train import init_model, train_model
from src.utils.init_datasets import init_datasets
from src.utils.utils import (load_yaml_config, reset_wandb_env,
                             update_nested_dict)

# helpful github issues for logging and sweeping k-fold cross-val with wandb:
# https://github.com/wandb/wandb/issues/5119
# https://github.com/wandb/wandb/issues/5003
# open issue: multi-gpu training with lightning + wandb sweep

os.environ["WANDB__SERVICE_WAIT"] = "300"


def outer_cv(train_config: dict, sweep_id: Optional[str] = None, verbose: bool = False):
    """Outer k-fold cross-validation loop.

    Args:
        train_config: Train configuration file.
        sweep_id: Optional sweep identity.
        verbose: True for print statements.
    """

    def cross_validate():
        """Runs actual k-fold cross-validation."""

        Path(train_config["output_dir"]).mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(train_config["output_dir"], "config.yaml"), "w"
        ) as outfile:
            yaml.dump(train_config, outfile, default_flow_style=False)

        config = OmegaConf.create(train_config)  # for dictionary dot notation
        dataset, test_dataset_swapped, test_dataset_balanced = init_datasets(config)

        num_val_samples = int(len(dataset) / config.num_folds)
        dataset_idxs = np.array(list(range(len(dataset))))

        group = config.model_name
        job_type = config.job_type

        if sweep_id is not None:
            group = group + "-" + sweep_id

            sweep_run = wandb.init(
                project=config.wandb_logger.project,
                group=group,
                job_type="average_runs",
                dir=config.output_dir,
            )
            merged_config = update_nested_dict(dict(config), wandb.config)
            config = OmegaConf.create(merged_config)
            sweep_run_name = sweep_run.name
            sweep_run.save()
            sweep_run_id = sweep_run.id
            sweep_run.finish()
            wandb.sdk.wandb_setup._setup(_reset=True)

            job_type = sweep_run_name
        else:
            sweep_run_name = None

        metrics = []
        for fold in range(config.num_folds):
            seed_everything(config.seed)
            if verbose:
                print("Processing fold: ", fold + 1)

            reset_wandb_env()

            val_idx = dataset_idxs[
                fold * num_val_samples : (fold + 1) * num_val_samples
            ]
            train_idx = np.concatenate(
                [
                    dataset_idxs[: fold * num_val_samples],
                    dataset_idxs[(fold + 1) * num_val_samples :],
                ],
                axis=0,
            )
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            if (config.data.dataset != "morpho_mnist") and (
                config.model.method == "rebalancing"
            ):
                labels = train_dataset.dataset._label_columns
                train_dataset.dataset.meta_labels_train = (
                    train_dataset.dataset._meta.iloc[train_idx][labels]
                )
                val_dataset.dataset.meta_labels_val = val_dataset.dataset._meta.iloc[
                    val_idx
                ][labels]

            if verbose:
                print(f"size train data {len(train_dataset)}")
                print(f"size val data {len(val_dataset)}")

            model = init_model(config)

            logger = WandbLogger(
                project=config.wandb_logger.project,
                group=group,
                job_type=job_type,
                name=job_type + f"-{fold}",
                save_dir=config.output_dir,
                config=dict(config),
                reinit=True,
            )
            checkpoint_filename = f"fold_{fold + 1}"
            if sweep_run_name is not None:
                checkpoint_filename = f"{sweep_run_name}-" + checkpoint_filename

            trainer = get_trainer(
                config,
                checkpoint_filename,
                logger,
            )
            trained_model = train_model(
                model,
                trainer,
                train_dataset,
                val_dataset,
                config,
            )
            test_c0_acc = test_model(
                trained_model,
                trainer,
                test_dataset_swapped,
                test_dataset_balanced,
                config,
                verbose=verbose,
            )
            wandb.finish()
            metrics.append(test_c0_acc)

        if sweep_id is not None:
            # resume the sweep run
            sweep_run = wandb.init(id=sweep_run_id, resume="must")
            # log metric to sweep run
            sweep_run.log({config.sweep_metric: sum(metrics) / len(metrics)})
            sweep_run.finish()

    return cross_validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tc",
        "--train_config",
        type=str,
        help="name of yaml config file",
        default="configs/morpho_mnist/train_dcor.yaml",
    )
    parser.add_argument(
        "-sid", "--sweep_id", type=str, help="wandb sweep id", default=None
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_config = load_yaml_config(config_filename=args.train_config)
    if args.sweep_id is not None:
        wandb.agent(
            sweep_id=args.sweep_id,
            function=outer_cv(train_config, sweep_id=args.sweep_id, verbose=True),
            project=train_config["wandb_logger"]["project"],
            count=1,
        )
        wandb.finish()
    else:
        outer_cv(train_config, sweep_id=None, verbose=True)()
