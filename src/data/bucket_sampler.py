import math
from typing import Iterator, Optional, Sequence

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import BatchSampler, Sampler


class WeightedBucketSampler(Sampler[int]):
    """Samples given bucket indices by given bucket weights.

    Attributes:
        buckets: Sequence of bucket indices.
        weights: Bucket sampling weights.
        shuffle: If True shuffle buckets.
        seed: Shuffle seed.
    """

    def __init__(
        self,
        buckets: Sequence[Tensor],
        weights: Sequence[float],
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        weights = torch.tensor([weight / sum(weights) for weight in weights])
        lengths = torch.tensor([len(bucket) for bucket in buckets])

        self.num_buckets = len(buckets)
        assert len(weights) == self.num_buckets
        self.buckets = buckets
        self.weights = weights

        self.shuffle = shuffle
        self.seed = seed

        lengths_per_weights = lengths / weights
        self.num_samples = int(torch.ceil(torch.max(lengths_per_weights)).item())
        self.divs, self.mods_floor, self.mods_fract = self.compute_divs_mods()
        num_deterministic_samples = sum(
            [
                div * len(bucket) + mod
                for bucket, div, mod in zip(self.buckets, self.divs, self.mods_floor)
            ]
        )
        self.num_missing_samples = self.num_samples - num_deterministic_samples

    def oversample_buckets(self):
        idxss = []
        if self.num_missing_samples > 0:
            extra_buckets = torch.multinomial(
                torch.tensor(self.mods_fract), self.num_missing_samples
            )
        for bucket_id, (bucket, div, mod) in enumerate(
            zip(self.buckets, self.divs, self.mods_floor)
        ):
            idxs = torch.cat([bucket for _ in range(div)])
            if self.num_missing_samples > 0:
                mod += sum(bucket_id == extra_buckets)
            if mod > 0:
                rest = bucket[
                    torch.multinomial(torch.ones_like(bucket, dtype=torch.float32), mod)
                ]
                idxs = torch.cat([idxs, rest])
            idxss.append(idxs)

        return torch.cat(idxss)

    def randperm_tensor(self, t) -> Tensor:
        # Deterministically shuffle based on seed.
        g = torch.Generator()
        g.manual_seed(self.seed)
        return t[torch.randperm(len(t), generator=g)]

    def __iter__(self) -> Iterator[int]:
        idxss = self.get_idxss()
        yield from iter(idxss)

    def get_idxss(self):
        idxss = self.oversample_buckets()
        if self.shuffle:
            idxss = self.randperm_tensor(idxss)

        return idxss.tolist()

    def __len__(self) -> int:
        return self.num_samples

    def compute_divs_mods(self):
        divs = []
        mods_floor = []
        mods_fract = []
        for bucket, weight in zip(self.buckets, self.weights):
            div, mod = np.divmod(self.num_samples * weight, len(bucket))
            mod_floor = int(mod)
            mod_fract = mod - mod_floor
            divs.append(int(div))
            mods_floor.append(mod_floor)
            mods_fract.append(mod_fract)
        return divs, mods_floor, mods_fract


def get_bucket_indices(buckets: Tensor, meta: Tensor):
    """Gets indices from buckets and meta.

    Args:
        buckets: Bucket label combinations.
        meta: Dataset labels.

    Returns:
        Sequence of bucket indices.
    """
    buckets = []
    for comb in buckets:
        bucket_mask = torch.all(meta == comb, dim=1)
        buckets.append(torch.argwhere(bucket_mask).squeeze(-1))
    return buckets


class DistributedBucketSampler(Sampler):
    """Sampler that restricts bucket sampler data loading to a subset of the dataset.

    Args:
        bucket_sampler: Sampler for weighted bucket sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    """

    def __init__(
        self,
        bucket_sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.bucket_sampler = bucket_sampler
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.num_samples_bucket_sampler = bucket_sampler.num_samples
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.num_samples_bucket_sampler % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.num_samples_bucket_sampler - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.num_samples_bucket_sampler / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        indices = self.bucket_sampler.get_idxss()
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DataPrepDDP(pl.LightningDataModule):
    """Manually define distributed training with own DDP sampler.

    Lightning is not able to make a distributed sampler automatically for custom samplers:
    github issue: https://github.com/Lightning-AI/pytorch-lightning/issues/5145.

    Attributes:
        config: Training yaml config.
        train_bucket_sampler: Train ``WeightedBucketSampler``.
        val_bucket_sampler: Val ``WeightedBucketSampler``.
        train_dataset: Training dataset.
        val_dataset: Validation dataset
    """

    def __init__(
        self,
        config,
        train_bucket_sampler,
        val_bucket_sampler,
        train_dataset,
        val_dataset,
    ):
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_bucket_sampler = train_bucket_sampler
        self.val_bucket_sampler = val_bucket_sampler

    def train_dataloader(self):
        train_distributed_subsampler = DistributedBucketSampler(
            self.train_bucket_sampler,
            drop_last=True,
        )

        train_batch_sampler = BatchSampler(
            sampler=train_distributed_subsampler,
            batch_size=self.config.data.batch_size,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=self.config.data.num_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )
        return train_loader

    def val_dataloader(self):
        val_distributed_subsampler = DistributedBucketSampler(
            self.val_bucket_sampler,
        )
        val_batch_sampler = BatchSampler(
            sampler=val_distributed_subsampler,
            batch_size=self.config.data.batch_size,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=self.config.data.num_workers,
            prefetch_factor=self.config.data.prefetch_factor,
        )
        return val_loader
