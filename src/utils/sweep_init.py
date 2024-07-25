import argparse
import os

import wandb

from src.utils.utils import load_yaml_config

os.environ["WANDB__SERVICE_WAIT"] = "300"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--wandb_project",
        type=str,
        default="medical-causal-disentanglement",
    )
    parser.add_argument(
        "-sc",
        "--sweep_config",
        type=str,
        help="name of yaml config file in the configs folder",
        default=None,
    )
    return parser.parse_args()


def sweep_init(project: str, sweep_config: dict):
    """Initializes hyperparameter sweep with wandb.

    Prints out the sweep id.

    Args:
        project: Wandb project name.
        sweep_config: Config file for hyperparameter sweep.
    """
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
    )
    print(sweep_id)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    sweep_config = load_yaml_config(config_filename=args.sweep_config)
    sweep_init(args.wandb_project, sweep_config)
