import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import Tensor

from src.data.bucket_sampler import get_bucket_indices


class CheXpert(torch.utils.data.Dataset):
    """Dataset for CheXpert chest radiography images.

    Source: https://stanfordmlgroup.github.io/competitions/chexpert/.

    Attributes:
        root: Data root directory.
        split: One of train, val or test.
        attribute_labels:
        frontal: If True filter out frontal views.
        image_size: Size images are resized and center cropped to.
        bucket_labels: Optional label co-occurences to sub-sample dataset.
        bucket_samples: Number of samples to take from each bucket.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        attribute_labels: List[str] = ["Pleural Effusion", "Sex"],
        frontal: bool = True,
        image_size: int = 320,
        bucket_labels: Optional[List[int]] = [[1, 1], [0, 1], [1, 0], [0, 0]],
        bucket_samples: Optional[List[int]] = [10400, 546, 546, 10400],
    ):
        self.root = root
        self.split = split
        self.attribute_labels = attribute_labels
        self.frontal = frontal

        self.bucket_labels = (
            torch.tensor(bucket_labels) if bucket_labels is not None else None
        )
        self.bucket_samples = bucket_samples

        self.image_size = image_size

        self._label_columns = []
        self._meta, self._labels = self._prepare_meta()
        self._data_files = self._meta["Path"].to_list()

    def __len__(self):
        return len(self._data_files)

    def _prepare_meta(self):
        # We use the first 1,222 patients (5,004 images) as test data.
        if self.split == "train":
            meta = pd.read_csv(
                os.path.join(self.root, "CheXpert-v1.0-small", f"{self.split}.csv")
            )
            meta = meta[5004:]
        elif self.split == "test":
            meta = pd.read_csv(
                os.path.join(self.root, "CheXpert-v1.0-small", "train.csv")
            )
            meta = meta[:5004]
        else:
            meta = pd.read_csv(
                os.path.join(self.root, "CheXpert-v1.0-small", f"{self.split}.csv")
            )
        meta_columns = ["Path", "Frontal/Lateral"] + self.attribute_labels
        meta = meta[meta_columns]

        # Filter meta.
        mask_filter_nans = meta.notnull().all(1)
        mask_filter_uncertain = (meta != -1).all(1)
        mask_filter_unknown = (meta != "Unknown").all(1)
        if self.frontal:
            mask_filter_frontal = meta["Frontal/Lateral"] == "Frontal"

        overall_mask = (
            mask_filter_nans
            & mask_filter_uncertain
            & mask_filter_unknown
            & mask_filter_frontal
        )
        meta = meta[overall_mask]

        # Categorize labels.
        for label in self.attribute_labels:
            categorical_label = meta[label].astype("category")
            meta[f"{label}_codes"] = categorical_label.cat.codes
            self._label_columns.append(f"{label}_codes")

        # Undersample buckets by number of bucket_samples.
        if (self.bucket_labels is not None) and (self.bucket_samples is not None):
            labels_tensor = torch.tensor(meta[self._label_columns].to_numpy())
            buckets_idxs = get_bucket_indices(self.bucket_labels, labels_tensor)
            idxs = []
            for bucket_idxs, samples in zip(buckets_idxs, self.bucket_samples):
                idxs.append(bucket_idxs[:samples])

            idxs = torch.cat(idxs)
            g = torch.Generator()
            g.manual_seed(42)
            idxs = idxs[torch.randperm(len(idxs), generator=g)]

            meta = meta.iloc[idxs]
        labels = torch.tensor(meta[self._label_columns].to_numpy().astype(np.int64))
        return meta, labels

    def __getitem__(self, index) -> Tuple[Tensor]:
        labels = self._labels[index]
        data_file = os.path.join(self.root, self._data_files[index])

        image = Image.open(data_file)
        image = torchvision.transforms.Resize(self.image_size)(image)
        image = torchvision.transforms.CenterCrop(self.image_size)(image)
        image = torchvision.transforms.ToTensor()(image)

        return image, labels
