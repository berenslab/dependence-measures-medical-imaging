import collections.abc
import os
from typing import List

import yaml


def load_yaml_config(config_filename: str):
    """Load yaml config.

    Args:
        config_filename: Filename to config.

    Returns:
        Loaded config.
    """
    with open(config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def update_nested_dict(d, u):
    """Update nested dictionary.

    Args:
        d: Dictionary.
        u: Dictionary to update d.

    Return:
        Merged/updated dictionaries.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def reset_wandb_env():
    """Reset wandb environment."""
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB__SERVICE_WAIT",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def optimizer_lr_scheduler(optimizer, p, initial_lr):
    """Adjust the learning rate of optimizer.
    Args:
        optimizer: Optimizer for updating parameters.
        p: A variable for adjusting learning rate.
        initial_lr: Initial learning rate at first step.
    Returns:
        Optimizer.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = initial_lr / ((1.0 + 10 * p) ** 0.75)

    return optimizer


def flatten_list(lsts: List[list]):
    """Flattents and unpacks list of lists to list of all the elements.

    Args:
        lsts: List of lists.

    Returns:
        Flattened list.
    """
    result = []
    for lst in lsts:
        result += lst
    return result
