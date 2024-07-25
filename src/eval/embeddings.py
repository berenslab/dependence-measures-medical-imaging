import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from src.train import init_model
from src.utils.init_datasets import init_datasets
from src.utils.utils import load_yaml_config

from matplotlib.colors import LinearSegmentedColormap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpts",
        "--ckpts_list",
        nargs="*",
        help="list of ckpts files",
        default=[
            "best_models/morpho_mnist/baseline/checkpoints/fold_1_epoch=97.ckpt",
            "best_models/morpho_mnist/rebalancing/checkpoints/fold_1_epoch=173.ckpt",
            "best_models/morpho_mnist/mine/checkpoints/fold_4_epoch=2999.ckpt",
            "best_models/morpho_mnist/dcor/checkpoints/fold_4_epoch=999.ckpt",
            "best_models/morpho_mnist/adv_cl/checkpoints/drawn-sweep-17-fold_3_epoch=803.ckpt",
        ],
    )
    parser.add_argument(
        "-cfgs",
        "--configs_list",
        nargs="*",
        help="list of config files folder",
        default=[
            "best_models/morpho_mnist/baseline/config.yaml",
            "best_models/morpho_mnist/rebalancing/config.yaml",
            "best_models/morpho_mnist/mine/config.yaml",
            "best_models/morpho_mnist/dcor/config.yaml",
            "best_models/morpho_mnist/adv_cl/config.yaml",
        ],
    )
    parser.add_argument(
        "-a",
        "--accelerator",
        type=str,
        help="compute device, either `cuda` or `cpu`",
        default="cuda",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=list,
        help="If you use `cuda`, choose number of devices",
        default=None,
    )
    return parser.parse_args()


def get_test_latents(
    config,
    model_ckpt,
    test_set,
    devices: Optional[List[int]] = [0],
    accelerator: str = "cuda",
):
    test_loader = torch.utils.data.DataLoader(
        test_set,
        128,
        shuffle=False,
        num_workers=config.test_data.num_workers,
        prefetch_factor=config.test_data.prefetch_factor,
        drop_last=False,
    )

    model = init_model(config)

    checkpoint = torch.load(model_ckpt, map_location=torch.device(accelerator))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    trainer = Trainer(
        devices=devices,
        accelerator=accelerator,
        strategy="auto",
    )
    latents = torch.concat(trainer.predict(model=model, dataloaders=test_loader))

    y1 = test_set._digit_labels
    y2 = test_set._pert_labels

    return latents, y1, y2


def embedding_figure(
    model_ckpts: List[str],
    configs: List[str],
    savefig_path: Optional[str] = "./figures/morpho_mnist_embeddings.pdf",
    fontsize: float = 8.5,
    accelerator: str = "cuda",
    devices: List[int] = [0],
):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(5, 0.8))
    colors= [(0.96, 0.06, 0.58), (0, 0.59, 0.8)]  # R -> G -> B
    custom_colors = LinearSegmentedColormap.from_list("custom_colors", colors, N=2)

    titles = [
        "Baseline",
        "Rebalance",
        "MINE",
        "dCor",
        "Adv. Classifier",
    ]

    for i, (ax, title) in enumerate(zip(axs.flatten(), titles)):
        config = load_yaml_config(config_filename=configs[i])
        config = OmegaConf.create(config)  # for dictionary dot notation

        _, _, test_dataset_balanced = init_datasets(config)
        print(f"len balanced test dataset: {len(test_dataset_balanced)}")

        print(f"Method: {config.model.method}.")
        print(model_ckpts[i])

        latents, y1, y2 = get_test_latents(
            config,
            model_ckpt=model_ckpts[i],
            test_set=test_dataset_balanced,
            accelerator=accelerator,
            devices=devices if accelerator == "cuda" else "auto",
        )

        random_idxs = np.array(range(latents.shape[0]))
        np.random.shuffle(random_idxs)
        y1 = y1[random_idxs]
        y2 = y2[random_idxs]

        if config.model.method != "adv_cl":
            z1 = latents[:, : config.model.subspace_dims[0]].numpy()
            z1 = z1[random_idxs]
            ax.scatter(
                z1[:, 0],
                z1[:, 1],
                c=y2,
                cmap=custom_colors,
                s=0.4,
            )
            ax.set_title(title, fontsize=fontsize)
            ax.axis("off")
        else:
            z = latents.numpy()
            z = z[random_idxs]

            scatter = ax.scatter(
                z[:, 0],
                z[:, 1],
                c=y2,
                cmap=custom_colors,
                s=0.4,
            )
            ax.set_title(title, fontsize=fontsize)
            ax.axis("off")
            handles, _ = scatter.legend_elements()
            ax.legend(
                handles=handles,
                labels=["thin", "thick"],
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                markerscale=0.3,
                title="writing style",
                fontsize=fontsize - 1,
                title_fontsize=fontsize,
                frameon=False,
            )

    if savefig_path is not None:
        output_file = Path(savefig_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            savefig_path,
            dpi=400,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )


if __name__ == "__main__":
    args = parse_args()

    font = {
        "family": "serif",
        "size": 12,
        "serif": "cmr10",
    }

    plt.rc("font", **font)
    plt.rcParams["text.usetex"] = True

    embedding_figure(
        model_ckpts=args.ckpts_list,
        configs=args.configs_list,
        savefig_path="./figures/morpho_mnist_embeddings.pdf",
        accelerator=args.accelerator,
        devices=args.devices,
    )
