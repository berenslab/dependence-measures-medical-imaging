import torch
from lightning.pytorch import Trainer

from src.data.bucket_sampler import (DataPrepDDP, WeightedBucketSampler,
                                     get_bucket_indices)
from src.models.encoder import EfficientNetB1, ResNetEncoder, SimpleEncoder
from src.models.method_abstraction import (MINE, AdversarialClassifierGRL,
                                           Baseline, dCor)
from src.models.module import SubspaceEncoderModule


def train_model(
    model: torch.nn.Module,
    trainer: Trainer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: dict,
):
    """Train the model with lightning.

    Args:
        model: Pytorch module to train.
        trainer: Lightning trainer module.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Config file.

    Returns:
        Trained model.
    """
    if (config.data.dataset != "morpho_mnist") and (
        config.model.method == "rebalancing"
    ):
        unique_bucket_labels = torch.tensor(config.data.bucket_labels)

        train_meta = torch.tensor(train_dataset.dataset.meta_labels_train.to_numpy())
        train_buckets = get_bucket_indices(unique_bucket_labels, train_meta)

        val_meta = torch.tensor(val_dataset.dataset.meta_labels_val.to_numpy())
        val_buckets = get_bucket_indices(unique_bucket_labels, val_meta)

        train_bucket_sampler = WeightedBucketSampler(
            buckets=train_buckets,
            weights=config.model.rebalancing_train_weights,
            shuffle=True,
        )

        print(f"train data loader samples: {train_bucket_sampler.__len__()}")

        val_bucket_sampler = WeightedBucketSampler(
            buckets=val_buckets,
            weights=config.model.rebalancing_val_weights,
            shuffle=False,
        )

        print(f"val loader samples: {val_bucket_sampler.__len__()}")

        # manually define distributed sampler
        # lightning is not able to make a distributed sampler automatically for custom samplers
        # github issue: https://github.com/Lightning-AI/pytorch-lightning/issues/5145
        if len(config.gpus) > 1:
            data_prep_ddp = DataPrepDDP(
                config,
                train_bucket_sampler,
                val_bucket_sampler,
                train_dataset,
                val_dataset,
            )
            trainer.fit(model, data_prep_ddp)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                config.data.batch_size,
                sampler=train_bucket_sampler,
                num_workers=config.data.num_workers,
                prefetch_factor=config.data.prefetch_factor,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                config.data.batch_size,
                sampler=val_bucket_sampler,
                num_workers=config.data.num_workers,
                prefetch_factor=config.data.prefetch_factor,
            )
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            prefetch_factor=config.data.prefetch_factor,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            prefetch_factor=config.data.prefetch_factor,
            drop_last=False,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model


def init_model(config):
    """Inits `SubspaceEncoderModule` with chosen method.

    Args:
        config: Configuration file.

    Returns:
        `SubspaceEncoderModule` as lightning module model.
    """
    if config.model.method == "mine":
        method = MINE(
            learning_rate_encoder=config.model.learning_rate_encoder,
            weight_decay_encoder=config.model.weight_decay_encoder,
            adam_w=config.model.adam_w,
            learning_rate_mine=config.model.learning_rate_mine,
            mine_batches_mi=config.model.mine_batches_mi,
            lambda_mi=config.model.lambda_dmeasure,
            subspace_dims=config.model.subspace_dims,
            class_dims=config.model.class_dims,
            two_layer_cs=config.model.two_layer_cs,
        )

    elif config.model.method == "dcor":
        method = dCor(
            learning_rate_encoder=config.model.learning_rate_encoder,
            weight_decay_encoder=config.model.weight_decay_encoder,
            adam_w=config.model.adam_w,
            lambda_dcor=config.model.lambda_dmeasure,
            subspace_dims=config.model.subspace_dims,
            class_dims=config.model.class_dims,
            two_layer_cs=config.model.two_layer_cs,
        )

    elif config.model.method == "adv_cl":
        method = AdversarialClassifierGRL(
            learning_rate_encoder=config.model.learning_rate_encoder,
            weight_decay_encoder=config.model.weight_decay_encoder,
            latent_dim=config.model.latent_dim,
            class_dims=config.model.class_dims,
            two_layer_cs=config.model.two_layer_cs,
            gamma=config.model.gamma,
            alpha_scale=config.model.alpha_scale,
        )
    elif (config.model.method == "baseline") or (config.model.method == "rebalancing"):
        method = Baseline(
            learning_rate_encoder=config.model.learning_rate_encoder,
            weight_decay_encoder=config.model.weight_decay_encoder,
            adam_w=config.model.adam_w,
            subspace_dims=config.model.subspace_dims,
            class_dims=config.model.class_dims,
            two_layer_cs=config.model.two_layer_cs,
        )

    if config.model.encoder == "resnet":
        encoder = ResNetEncoder(
            resnet_backbone=config.model.resnet_backbone,
            latent_dim=config.model.latent_dim,
            in_channels=config.model.in_channels,
        )
    elif config.model.encoder == "efficientnet_b1":
        encoder = EfficientNetB1(
            latent_dim=config.model.latent_dim,
            in_channels=config.model.in_channels,
        )
    elif config.model.encoder == "simple_encoder":
        encoder = SimpleEncoder(
            latent_dim=config.model.latent_dim,
        )

    model = SubspaceEncoderModule(
        method=method,
        encoder=encoder,
        class_dims=config.model.class_dims,
        warmup_epochs=config.model.warmup_epochs,
        monitor_metric=config.monitor_metric,
        save_after_x_epochs=config.save_after_x_epochs,
        save_only_last_epoch=config.save_only_last_epoch,
        n_test=2,
    )
    return model
