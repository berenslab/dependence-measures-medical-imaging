import torch
import lightning

def test_model(
    model: torch.nn.Module,
    trainer: lightning.pytorch.Trainer,
    test_dataset_swapped: torch.utils.data.Dataset,
    test_dataset_balanced: torch.utils.data.Dataset,
    config: dict,
    verbose: bool = True,
):
    """Tests the model with lightning trainer.

    Args:
        model: Trained model.
        trainer: Lightning trainer module.
        test_dataset_swapped: Inverted test distribution
            (to the training correlation).
        test_dataset_balanced: Balances test distribution.
        config: Config file.
        verbose: If True print out statements.

    Returns:
        Sweep metric on inverted test distribution.
    """
    if verbose:
        print(f"size test data swapped {len(test_dataset_swapped)}")
        print(f"size test data balanced {len(test_dataset_balanced)}")

    test_loader_swapped = torch.utils.data.DataLoader(
        test_dataset_swapped,
        config.test_data.batch_size,
        shuffle=False,
        num_workers=config.test_data.num_workers,
        prefetch_factor=config.test_data.prefetch_factor,
        drop_last=False,
        pin_memory=True,
    )
    test_loader_balanced = torch.utils.data.DataLoader(
        test_dataset_balanced,
        config.test_data.batch_size,
        shuffle=False,
        num_workers=config.test_data.num_workers,
        prefetch_factor=config.test_data.prefetch_factor,
        drop_last=False,
        pin_memory=True,
    )

    metrics = trainer.test(
        model,
        dataloaders=[test_loader_swapped, test_loader_balanced],
        ckpt_path="best",
    )
    return metrics[0][config.sweep_metric]
