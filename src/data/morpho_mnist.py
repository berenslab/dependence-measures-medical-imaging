import gzip
import os
import struct
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor


class MorphoMNISTCorrelated(torch.utils.data.Dataset):
    """Dataset that correlates Morpho-MNIST `global` images.

    Source: https://github.com/dccastro/Morpho-MNIST.

    Attributes:
        root: Root directory of dataset where `train-images-idx3-ubyte`
            and  `t10k-images-idx3-ubyte` exist.
        train: If True, creates dataset from `train-images-idx3-ubyte`,
            otherwise from `t10k-images-idx3-ubyte`.
        transform: A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform: A function/transform that takes in the target and
            transforms it.
        correlation_strength: Correlation strength from thin/small and thick/large
            versus thin/large and thick/small.
        rebalancing: If True, rebalance the correlation by over-sampling underrepresented groups.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        correlation_strength: List[int] = [95, 5],
        rebalancing: bool = False,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.rebalancing = rebalancing
        if train:
            images = load_idx(os.path.join(root, "train-images-idx3-ubyte.gz"))
            digit_labels = load_idx(os.path.join(root, "train-labels-idx1-ubyte.gz"))
            pert_labels = load_idx(os.path.join(root, "train-pert-idx1-ubyte.gz"))
        else:
            images = load_idx(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
            digit_labels = load_idx(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
            pert_labels = load_idx(os.path.join(root, "t10k-pert-idx1-ubyte.gz"))

        (
            images_thin_small,
            digit_labels_thin_small,
            pert_labels_thin_small,
        ) = morpho_mnist_subset(
            images,
            digit_labels,
            pert_labels,
            thin=True,
            digits="small",
        )
        (
            images_thin_large,
            digit_labels_thin_large,
            pert_labels_thin_large,
        ) = morpho_mnist_subset(
            images,
            digit_labels,
            pert_labels,
            thin=True,
            digits="large",
        )
        (
            images_thick_large,
            digit_labels_thick_large,
            pert_labels_thick_large,
        ) = morpho_mnist_subset(
            images,
            digit_labels,
            pert_labels,
            thin=False,
            digits="large",
        )
        (
            images_thick_small,
            digit_labels_thick_small,
            pert_labels_thick_small,
        ) = morpho_mnist_subset(
            images,
            digit_labels,
            pert_labels,
            thin=False,
            digits="small",
        )

        # Numbers for co-occurence matrix:
        # diagonals: thin/small, thick/large, off-diagonals: thin/large, thick/small
        num_images = min(images_thin_small.shape[0], images_thick_large.shape[0])
        num_images_diagonals = int(num_images * correlation_strength[0] / 100)
        if correlation_strength[0] + correlation_strength[1] != 100:
            raise AssertionError(
                "We assume that the sum correlation strengths of main-diagonals and off-diagonals sum to 100."
            )
        num_images_off_diagonals = num_images - num_images_diagonals

        # diagonals
        images_thin_small = images_thin_small[:num_images_diagonals]
        digit_labels_thin_small = digit_labels_thin_small[:num_images_diagonals]
        pert_labels_thin_small = pert_labels_thin_small[:num_images_diagonals]

        images_thick_large = images_thick_large[:num_images_diagonals]
        digit_labels_thick_large = digit_labels_thick_large[:num_images_diagonals]
        pert_labels_thick_large = pert_labels_thick_large[:num_images_diagonals]

        # off-diagonals
        images_thin_large = images_thin_large[:num_images_off_diagonals]
        digit_labels_thin_large = digit_labels_thin_large[:num_images_off_diagonals]
        pert_labels_thin_large = pert_labels_thin_large[:num_images_off_diagonals]

        images_thick_small = images_thick_small[:num_images_off_diagonals]
        digit_labels_thick_small = digit_labels_thick_small[:num_images_off_diagonals]
        pert_labels_thick_small = pert_labels_thick_small[:num_images_off_diagonals]

        # rebalancing: oversample off-diagonals
        if self.rebalancing:
            if correlation_strength[0] < correlation_strength[1]:
                raise AssertionError(
                    "Rebalancing is only implemented for over-represented main diagionals."
                )
            num_repeat = num_images_diagonals // num_images_off_diagonals
            rest = num_images_diagonals % num_images_off_diagonals
            random_rest = np.random.choice(
                np.array(list(range(num_images_off_diagonals))), size=rest
            )

            rebalance_data = lambda data: np.concatenate(
                [
                    np.repeat(data, num_repeat, axis=0),
                    data[random_rest],
                ],
                axis=0,
            )

            # off-diagonals
            images_thin_large = rebalance_data(images_thin_large)
            digit_labels_thin_large = rebalance_data(digit_labels_thin_large)
            pert_labels_thin_large = rebalance_data(pert_labels_thin_large)

            images_thick_small = rebalance_data(images_thick_small)
            digit_labels_thick_small = rebalance_data(digit_labels_thick_small)
            pert_labels_thick_small = rebalance_data(pert_labels_thick_small)

        self._images = np.concatenate(
            [
                images_thin_small,
                images_thick_large,
                images_thin_large,
                images_thick_small,
            ],
            axis=0,
        )
        self._digit_labels = np.concatenate(
            [
                digit_labels_thin_small,
                digit_labels_thick_large,
                digit_labels_thin_large,
                digit_labels_thick_small,
            ],
            axis=0,
        )
        self._pert_labels = np.concatenate(
            [
                pert_labels_thin_small,
                pert_labels_thick_large,
                pert_labels_thin_large,
                pert_labels_thick_small,
            ],
            axis=0,
        )

        # Change pert labels from [1,2] to [0,1].
        map_labels = {1: 0, 2: 1}
        self._pert_labels = np.array([map_labels[label] for label in self._pert_labels])

        # Change digit labels from [0-9] to [0,1] for small vs large.
        map_digit_labels = {digit: 0 if digit < 5 else 1 for digit in range(10)}
        self._digit_labels = np.array(
            [map_digit_labels[label] for label in self._digit_labels]
        )

        self._len_datatset = self._images.shape[0]

        # randomly shuffle data
        rand_perm = np.random.default_rng(seed=42).permutation(self._len_datatset)
        self._images = self._images[rand_perm]
        self._digit_labels = self._digit_labels[rand_perm]
        self._pert_labels = self._pert_labels[rand_perm]

    def __len__(self):
        return self._len_datatset

    def __getitem__(self, index) -> Tuple[Tensor]:
        image, digit_label, pert_label = (
            self._images[index],
            self._digit_labels[index],
            self._pert_labels[index],
        )

        image = Image.fromarray(image, mode="L")

        if self.transform is not None:
            image = self.transform(image)

        image = torchvision.transforms.ToTensor()(image)

        if self.target_transform is not None:
            digit_label = self.target_transform(digit_label)
            pert_label = self.target_transform(pert_label)

        return image, torch.tensor([digit_label, pert_label])


def morpho_mnist_subset(
    images,
    digit_labels,
    pert_labels,
    thin: bool = True,
    digits: str = "small",
):
    if thin:
        pert_label = 1
    else:  # thick
        pert_label = 2
    if digits == "small":
        digit_mask = digit_labels < 5
    elif digits == "large":
        digit_mask = digit_labels > 4
    mask = (pert_labels == pert_label) & digit_mask
    return images[mask], digit_labels[mask], pert_labels[mask]


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Args:
        path: Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns: Output array of dtype `uint8`.

    References:
        http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data
