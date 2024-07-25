from typing import Tuple

import torch

from src.data.chexpert import CheXpert
from src.data.morpho_mnist import MorphoMNISTCorrelated


def init_datasets(config) -> Tuple[torch.utils.data.Dataset]:
    """Initialize all the datasets from config

    Datasets: training (k-fold) and the two shifted test distributions

    Args:
        config: Configuration file.

    Returns:
        Tuple of three datasets.
    """

    if config.data.dataset == "morpho_mnist":
        if config.model.method == "rebalancing":
            rebalancing = True
        else:
            rebalancing = False

        train_dataset = MorphoMNISTCorrelated(
            root=config.data.dataset_path,
            train=True,
            correlation_strength=config.data.correlation_strength,
            rebalancing=rebalancing,
        )
        test_dataset_swapped = MorphoMNISTCorrelated(
            root=config.data.dataset_path,
            train=False,
            correlation_strength=config.test_data.correlation_strength_swapped,
        )
        test_dataset_balanced = MorphoMNISTCorrelated(
            root=config.data.dataset_path,
            train=False,
            correlation_strength=config.test_data.correlation_strength_balanced,
        )
        return train_dataset, test_dataset_swapped, test_dataset_balanced

    elif config.data.dataset == "chexpert":
        train_dataset = CheXpert(
            root=config.data.dataset_path,
            split="train",
            attribute_labels=config.data.attribute_labels,
            frontal=config.data.frontal,
            image_size=config.data.image_size,
            bucket_labels=config.data.bucket_labels,
            bucket_samples=config.data.bucket_samples,
        )
        test_dataset_swapped = CheXpert(
            root=config.data.dataset_path,
            split="test",
            attribute_labels=config.data.attribute_labels,
            frontal=config.data.frontal,
            image_size=config.data.image_size,
            bucket_labels=config.data.bucket_labels,
            bucket_samples=config.test_data.bucket_samples_swapped,
        )
        test_dataset_balanced = CheXpert(
            root=config.data.dataset_path,
            split="test",
            attribute_labels=config.data.attribute_labels,
            frontal=config.data.frontal,
            image_size=config.data.image_size,
            bucket_labels=config.data.bucket_labels,
            bucket_samples=config.test_data.bucket_samples_balanced,
        )
        return train_dataset, test_dataset_swapped, test_dataset_balanced

    else:
        AssertionError(
            f"The init for the dataset {config.data.dataset} is not implemented."
        )
