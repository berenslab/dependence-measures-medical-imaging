import argparse
import os

import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsClassifier

from src.train import init_model
from src.utils.utils import load_yaml_config
from src.utils.init_datasets import init_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="name of yaml config file",
        default="best_models/morpho_mnist/baseline/config.yaml",
    )
    parser.add_argument(
        "-ckpts",
        "--ckpts_folder",
        type=str,
        help="path to ckpts folder",
        default="best_models/morpho_mnist/baseline/checkpoints",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        help="number of neighbors for kNN classifier",
        default=30,
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
        default=[0],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.config)
    config = OmegaConf.create(config)  # for dictionary dot notation

    train_dataset, _, test_dataset_balanced = init_datasets(config)
    print(f"len train dataset: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        config.test_data.batch_size,
        shuffle=False,
        num_workers=config.test_data.num_workers,
        prefetch_factor=config.test_data.prefetch_factor,
        drop_last=False,
    )
    print(f"len balanced test dataset: {len(test_dataset_balanced)}")
    test_loader_balanced = torch.utils.data.DataLoader(
        test_dataset_balanced,
        config.test_data.batch_size,
        shuffle=False,
        num_workers=config.test_data.num_workers,
        prefetch_factor=config.test_data.prefetch_factor,
        drop_last=False,
    )

    # Compute mean kNN classifier accuracy
    k = args.k
    model = init_model(config)

    z1y1_accs = []
    z1y2_accs = []
    z2y1_accs = []
    z2y2_accs = []

    zy1_accs = []
    zy2_accs = []

    print(f"Method: {config.model.method}.")

    model_ckpts = os.listdir(args.ckpts_folder)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    for model_ckpt in model_ckpts:
        model_ckpt = os.path.join(args.ckpts_folder, model_ckpt)

        checkpoint = torch.load(model_ckpt, map_location=torch.device(args.accelerator))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        trainer = Trainer(
            devices=args.devices if args.accelerator == "cuda" else "auto",
            accelerator=args.accelerator,
            strategy="auto",
        )
        
        train_latents = torch.concat(
            trainer.predict(model=model, dataloaders=train_loader)
        )
        test_latents = torch.concat(
            trainer.predict(model=model, dataloaders=test_loader_balanced)
        )

        if config.model.method != "adv_cl":
            z1_train = train_latents[:, : config.model.subspace_dims[0]].numpy()
            z2_train = train_latents[:, config.model.subspace_dims[0] :].numpy()

            y1_train = train_dataset._digit_labels
            y2_train = train_dataset._pert_labels

            z1_test = test_latents[:, : config.model.subspace_dims[0]].numpy()
            z2_test = test_latents[:, config.model.subspace_dims[0] :].numpy()

            y1_test = test_dataset_balanced._digit_labels
            y2_test = test_dataset_balanced._pert_labels

            knn_classifier.fit(z1_train, y1_train)
            z1y1_accs.append(knn_classifier.score(z1_test, y1_test))

            knn_classifier.fit(z1_train, y2_train)
            z1y2_accs.append(knn_classifier.score(z1_test, y2_test))

            knn_classifier.fit(z2_train, y1_train)
            z2y1_accs.append(knn_classifier.score(z2_test, y1_test))

            knn_classifier.fit(z2_train, y2_train)
            z2y2_accs.append(knn_classifier.score(z2_test, y2_test))

        else:
            z_train = train_latents.numpy()

            y1_train = train_dataset._digit_labels
            y2_train = train_dataset._pert_labels

            z_test = test_latents.numpy()

            y1_test = test_dataset_balanced._digit_labels
            y2_test = test_dataset_balanced._pert_labels

            knn_classifier.fit(z_train, y1_train)
            zy1_accs.append(knn_classifier.score(z_test, y1_test))

            knn_classifier.fit(z_train, y2_train)
            zy2_accs.append(knn_classifier.score(z_test, y2_test))

    if config.model.method != "adv_cl":
        print("Average accuracy scores on the balanced test set:")
        print(
            f"z1y1: {np.array(z1y1_accs).mean():0.3f}, z2y1: {np.array(z2y1_accs).mean():0.3f}"
        )
        print(
            f"z1y2: {np.array(z1y2_accs).mean():0.3f}, z2y2: {np.array(z2y2_accs).mean():0.3f}"
        )
    else:
        print("Average accuracy scores on the balanced test set:")
        print(f"zy1: {np.array(zy1_accs).mean():0.3f}")
        print(f"zy2: {np.array(zy2_accs).mean():0.3f}")
