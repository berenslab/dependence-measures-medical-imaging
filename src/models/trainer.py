import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


def get_trainer(
    config,
    checkpoint_filename,
    logger,
):
    """Create lightning trainer.

    Args:
        config: Configuration file.
        checkpoint_filename: Checkpoint directory.
        logger: Logger, e.g. tensorboard or wandb.

    Returns:
        Lightning trainer
    """
    seed_everything(config.seed)
    if checkpoint_filename is not None:
        filename = checkpoint_filename + "_{epoch}"
    else:
        filename = "_{epoch}"

    if config.save_only_last_epoch:
        checkpoint_callback = ModelCheckpoint(
            monitor=None,
            dirpath=os.path.join(config.output_dir, "checkpoints"),
            filename=filename,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=config.save_top_k,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            dirpath=os.path.join(config.output_dir, "checkpoints"),
            filename=filename,
        )

    if (config.model.method == "rebalancing") and (len(config.gpus) > 1):
        use_distributed_sampler = True
    else:
        use_distributed_sampler = False
    trainer = Trainer(
        devices=config.gpus,
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true" if len(config.gpus) > 1 else "auto",
        use_distributed_sampler=use_distributed_sampler,
        num_sanity_val_steps=config.sanity_steps,
        max_epochs=config.max_epoch,
        limit_val_batches=config.val_check_percent,
        callbacks=[checkpoint_callback],
        val_check_interval=float(min(config.val_check_interval, 1)),
        check_val_every_n_epoch=max(1, config.val_check_interval),
        logger=logger,
        benchmark=True,
    )
    return trainer
