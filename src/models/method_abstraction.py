from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import Tensor

from src.models.classification_heads import (LinearClassifier,
                                             TwoLayerLinearClassifier)
from src.models.dependence_measures import (AdvClassifier, MIEstimator,
                                            distance_correlation)
from src.utils.utils import flatten_list, optimizer_lr_scheduler


class SubspaceDistanglementMethod(ABC, torch.nn.Module):
    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def gradient_step(self):
        pass

    @abstractmethod
    def get_classifiers(self):
        pass

    @abstractmethod
    def predict(self, z) -> List[Tensor]:
        pass

    @abstractmethod
    def get_losses(self):
        pass


class Baseline(SubspaceDistanglementMethod):
    """Baseline method, only classification heads, no dependence measure minimization.

    Attributes:
        learning_rate_encoder: Learning rate for encoder optimizer.
        weight_decay_encoder: Weight decay for encoder.
        adam_w: If True use AdamW optimizer.
        subspace_dims: number of subspace dimensions.
        class_dims: Number of classes the classification heads map to.
        two_layer_cs: If True use `TwoLayerLinearClassifier`.
    """

    def __init__(
        self,
        learning_rate_encoder: float,
        weight_decay_encoder: float,
        adam_w: bool,
        subspace_dims: List[int],
        class_dims: List[int],
        two_layer_cs: bool,
    ):
        super().__init__()
        self.learning_rate_encoder = learning_rate_encoder
        self.weight_decay_encoder = weight_decay_encoder
        self.adam_w = adam_w

        self.subspace_dims = subspace_dims
        self.class_dims = class_dims
        self.two_layer_cs = two_layer_cs

        self.classifiers = self.get_classifiers()

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        encoder_params = list(encoder.parameters()) + flatten_list(
            classifier_parameters
        )
        if self.adam_w:
            encoder_opt = torch.optim.AdamW(
                encoder_params,
                lr=self.learning_rate_encoder,
                weight_decay=self.weight_decay_encoder,
            )
        else:
            encoder_opt = torch.optim.Adam(
                encoder_params,
                lr=self.learning_rate_encoder,
                weight_decay=self.weight_decay_encoder,
            )
        return encoder_opt

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self.compute_average_ce_loss(logits, labels)
        overall_loss = ce_loss

        # Update encoder model.
        encoder_opt.zero_grad()
        lightning_module.manual_backward(overall_loss)
        encoder_opt.step()

        return ce_loss, torch.tensor(0.0), overall_loss, logits

    def predict(
        self,
        z,
    ):
        y_hat_logits_list = []
        for i in range(len(self.subspace_dims)):
            subspace = z[
                :, sum(self.subspace_dims[:i]) : sum(self.subspace_dims[: i + 1])
            ]
            y_hat_logits = self.classifiers[i](subspace).squeeze()
            y_hat_logits_list.append(y_hat_logits)

        return y_hat_logits_list

    def get_classifiers(self):
        num_subspaces = len(self.subspace_dims)
        Cs = []
        for i in range(num_subspaces):
            if self.two_layer_cs:
                Cs.append(
                    TwoLayerLinearClassifier(
                        z_shape=self.subspace_dims[i],
                        c_shape=self.class_dims[i],
                    )
                )
            else:
                Cs.append(
                    LinearClassifier(
                        z_shape=self.subspace_dims[i],
                        c_shape=self.class_dims[i],
                    )
                )
        return torch.nn.ModuleList(Cs)

    def compute_average_ce_loss(self, logits, labels):
        ce_loss = 0.0
        for i, subspace_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.class_dims[i] == 1:
                subspace_logits = subspace_logits.squeeze()
                subspace_labels = subspace_labels.to(torch.float32)
                subspace_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    subspace_logits, subspace_labels
                )
            else:
                subspace_loss = torch.nn.functional.cross_entropy(
                    subspace_logits, subspace_labels
                )
            ce_loss = ce_loss + subspace_loss
        return ce_loss / len(logits)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self.compute_average_ce_loss(logits, labels)
        overall_loss = ce_loss
        return logits, ce_loss, torch.tensor(0.0), overall_loss


class MINE(SubspaceDistanglementMethod):
    """MINE method, classification heads, MI minimization.

    Attributes:
        learning_rate_encoder: Learning rate for encoder optimizer.
        weight_decay_encoder: Weight decay for encoder.
        learning_rate_mine: Learning rate for MI estimator network.
        mine_batches_mi: MI estimator network is optimizes for `mine_batches_mi-1`,
            before the encoder is optimized for one step.
        adam_w: If True use AdamW optimizer.
        lambda_mi: Weight for MI minimization.
        subspace_dims: number of subspace dimensions.
        class_dims: Number of classes the classification heads map to.
        two_layer_cs: If True use `TwoLayerLinearClassifier`.
    """

    def __init__(
        self,
        learning_rate_encoder: float,
        weight_decay_encoder: float,
        learning_rate_mine: float,
        mine_batches_mi: float,
        adam_w: bool,
        lambda_mi: float,
        subspace_dims: List[int],
        class_dims: List[int],
        two_layer_cs: bool,
    ):
        super().__init__()
        self.learning_rate_encoder = learning_rate_encoder
        self.weight_decay_encoder = weight_decay_encoder

        self.learning_rate_mine = learning_rate_mine
        self.mine_batches_mi = mine_batches_mi
        self.lambda_mi = lambda_mi

        self.subspace_dims = subspace_dims
        self.class_dims = class_dims
        self.two_layer_cs = two_layer_cs
        self.classifiers = self.get_classifiers()

        self.adam_w = adam_w

        self.mi_estimator = MIEstimator(
            feature_dim=sum(subspace_dims),
        )

    def get_classifiers(self):
        num_subspaces = len(self.subspace_dims)
        Cs = []
        for i in range(num_subspaces):
            if self.two_layer_cs:
                Cs.append(
                    TwoLayerLinearClassifier(
                        z_shape=self.subspace_dims[i],
                        c_shape=self.class_dims[i],
                    )
                )
            else:
                Cs.append(
                    LinearClassifier(
                        z_shape=self.subspace_dims[i],
                        c_shape=self.class_dims[i],
                    )
                )
        return torch.nn.ModuleList(Cs)

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        encoder_params = list(encoder.parameters()) + flatten_list(
            classifier_parameters
        )
        if self.adam_w:
            encoder_opt = torch.optim.AdamW(
                encoder_params,
                lr=self.learning_rate_encoder,
                weight_decay=self.weight_decay_encoder,
            )
        else:
            encoder_opt = torch.optim.Adam(
                encoder_params,
                lr=self.learning_rate_encoder,
                weight_decay=self.weight_decay_encoder,
            )

        dmeasure_opt = torch.optim.Adam(
            self.mi_estimator.parameters(),
            lr=self.learning_rate_mine,
        )
        return encoder_opt, dmeasure_opt

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt, dmeasure_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self.compute_average_ce_loss(logits, labels)

        mi_estimator_update = batch_idx % self.mine_batches_mi != 0
        mi_estimate = self.estimate_mi(z)

        if mi_estimator_update:
            # Update mi estimation model.
            dmeasure_opt.zero_grad()
            lightning_module.manual_backward(-mi_estimate)
            dmeasure_opt.step()
            overall_loss = ce_loss + self.lambda_mi * mi_estimate

        else:
            # Update encoder model.
            overall_loss = ce_loss
            if not warmup:
                overall_loss = overall_loss + self.lambda_mi * mi_estimate

            encoder_opt.zero_grad()
            lightning_module.manual_backward(overall_loss)
            encoder_opt.step()

        return ce_loss, mi_estimate, overall_loss, logits

    def estimate_mi(self, z):
        z1 = z[:, 0 : self.subspace_dims[0]]
        z2 = z[
            :,
            self.subspace_dims[0] : self.subspace_dims[0] + self.subspace_dims[1],
        ]
        return self.mi_estimator(z1, z2)

    def predict(
        self,
        z,
    ):
        y_hat_logits_list = []
        for i in range(len(self.subspace_dims)):
            subspace = z[
                :, sum(self.subspace_dims[:i]) : sum(self.subspace_dims[: i + 1])
            ]
            y_hat_logits = self.classifiers[i](subspace).squeeze()
            y_hat_logits_list.append(y_hat_logits)

        return y_hat_logits_list

    def compute_average_ce_loss(self, logits, labels):
        ce_loss = 0.0
        for i, subspace_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.class_dims[i] == 1:
                subspace_logits = subspace_logits.squeeze()
                subspace_labels = subspace_labels.to(torch.float32)
                subspace_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    subspace_logits, subspace_labels
                )
            else:
                subspace_loss = torch.nn.functional.cross_entropy(
                    subspace_logits, subspace_labels
                )
            ce_loss = ce_loss + subspace_loss
        return ce_loss / len(logits)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self.compute_average_ce_loss(logits, labels)
        mi_estimate = torch.clamp(self.estimate_mi(z), min=0)
        overall_loss = ce_loss + self.lambda_mi * mi_estimate
        return logits, ce_loss, mi_estimate, overall_loss


class dCor(SubspaceDistanglementMethod):
    """dCor method, classification heads, dcor minimization.

    Attributes:
        learning_rate_encoder: Learning rate for encoder optimizer.
        weight_decay_encoder: Weight decay for encoder.
        adam_w: If True use AdamW optimizer.
        lambda_dcor: Weight for dCor minimization.
        subspace_dims: number of subspace dimensions.
        class_dims: Number of classes the classification heads map to.
        two_layer_cs: If True use `TwoLayerLinearClassifier`.
    """

    def __init__(
        self,
        learning_rate_encoder: float,
        weight_decay_encoder: float,
        adam_w: bool,
        lambda_dcor: float,
        subspace_dims: List[int],
        class_dims: List[int],
        two_layer_cs: bool,
    ):
        super().__init__()
        self.learning_rate_encoder = learning_rate_encoder
        self.weight_decay_encoder = weight_decay_encoder

        self.lambda_dcor = lambda_dcor

        self.subspace_dims = subspace_dims
        self.class_dims = class_dims
        self.two_layer_cs = two_layer_cs

        self.classifiers = self.get_classifiers()

        self.adam_w = adam_w

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        encoder_params = list(encoder.parameters()) + flatten_list(
            classifier_parameters
        )
        if self.adam_w:
            encoder_opt = torch.optim.AdamW(
                encoder_params,
                lr=self.learning_rate_encoder,
                weight_decay=self.weight_decay_encoder,
            )
        else:
            encoder_opt = torch.optim.Adam(
                encoder_params,
                lr=self.learning_rate_encoder,
                weight_decay=self.weight_decay_encoder,
            )
        return encoder_opt

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self.compute_average_ce_loss(logits, labels)

        # Update encoder model.
        overall_loss = ce_loss
        dcor_estimate = self.estimate_dcor(z)
        if not warmup:
            overall_loss = overall_loss + self.lambda_dcor * dcor_estimate

        encoder_opt.zero_grad()
        lightning_module.manual_backward(overall_loss)
        encoder_opt.step()

        return ce_loss, dcor_estimate, overall_loss, logits

    def get_classifiers(self):
        num_subspaces = len(self.subspace_dims)
        Cs = []
        for i in range(num_subspaces):
            if self.two_layer_cs:
                Cs.append(
                    TwoLayerLinearClassifier(
                        z_shape=self.subspace_dims[i],
                        c_shape=self.class_dims[i],
                    )
                )
            else:
                Cs.append(
                    LinearClassifier(
                        z_shape=self.subspace_dims[i],
                        c_shape=self.class_dims[i],
                    )
                )
        return torch.nn.ModuleList(Cs)

    def estimate_dcor(
        self,
        z,
    ):
        z1 = z[:, : self.subspace_dims[0]]
        z2 = z[:, self.subspace_dims[0] : sum(self.subspace_dims)]
        return distance_correlation(z1, z2)

    def predict(
        self,
        z,
    ):
        y_hat_logits_list = []
        for i in range(len(self.subspace_dims)):
            subspace = z[
                :, sum(self.subspace_dims[:i]) : sum(self.subspace_dims[: i + 1])
            ]
            y_hat_logits = self.classifiers[i](subspace).squeeze()
            y_hat_logits_list.append(y_hat_logits)

        return y_hat_logits_list

    def compute_average_ce_loss(self, logits, labels):
        ce_loss = 0.0
        for i, subspace_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.class_dims[i] == 1:
                subspace_logits = subspace_logits.squeeze()
                subspace_labels = subspace_labels.to(torch.float32)
                subspace_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    subspace_logits, subspace_labels
                )
            else:
                subspace_loss = torch.nn.functional.cross_entropy(
                    subspace_logits, subspace_labels
                )
            ce_loss = ce_loss + subspace_loss
        return ce_loss / len(logits)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self.compute_average_ce_loss(logits, labels)
        dcor_estimate = self.estimate_dcor(z)
        overall_loss = ce_loss + self.lambda_dcor * dcor_estimate
        return logits, ce_loss, dcor_estimate, overall_loss


class AdversarialClassifierGRL(SubspaceDistanglementMethod):
    """Adversarial classifier method, classification heads, CE loss maximization.

    Attributes:
        learning_rate_encoder: Learning rate for encoder optimizer.
        weight_decay_encoder: Weight decay for encoder.
        latent_dim: Latent space dimensions.
        class_dims: Number of classes the classification heads map to.
        two_layer_cs: If True use `TwoLayerLinearClassifier`.
        gamma: Hyperparamater for lambda schedule.
        alpha_scale: Hyperparamater for lambda schedule.
    """

    def __init__(
        self,
        learning_rate_encoder: float,
        weight_decay_encoder: float,
        latent_dim: int,
        class_dims: List[int],
        two_layer_cs: bool,
        gamma: float = 10.0,
        alpha_scale: float = 1,
    ):
        super().__init__()
        self.learning_rate_encoder = learning_rate_encoder
        self.weight_decay_encoder = weight_decay_encoder

        self.latent_dim = latent_dim
        self.class_dims = class_dims
        self.two_layer_cs = two_layer_cs
        self.gamma = gamma
        self.alpha_scale = alpha_scale

        self.classifiers = self.get_classifiers()

    def get_classifiers(self):
        if self.two_layer_cs:
            self.classifier = TwoLayerLinearClassifier(
                z_shape=self.latent_dim,
                c_shape=self.class_dims[0],
            )
        else:
            self.classifier = LinearClassifier(
                z_shape=self.latent_dim,
                c_shape=self.class_dims[0],
            )
        self.adv_classifier = AdvClassifier(
            z_shape=self.latent_dim,
            c_shape=self.class_dims[1],
        )
        return torch.nn.ModuleList([self.classifier, self.adv_classifier])

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        optimizer = torch.optim.SGD(
            list(encoder.parameters()) + flatten_list(classifier_parameters),
            lr=self.learning_rate_encoder,
            weight_decay=self.weight_decay_encoder,
            momentum=0.9,
        )
        return optimizer

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()
        encoder_opt = optimizer_lr_scheduler(encoder_opt, p, self.learning_rate_encoder)

        alpha = (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1) * self.alpha_scale

        logits = self.predict(z, alpha)
        ce_loss_cl, ce_loss_adv_cl = self.compute_ce_losses(logits, labels)

        overall_loss = ce_loss_cl + ce_loss_adv_cl

        encoder_opt.zero_grad()
        lightning_module.manual_backward(overall_loss)
        encoder_opt.step()

        return ce_loss_cl, ce_loss_adv_cl, overall_loss, logits

    def predict(
        self,
        z,
        alpha=None,
    ):
        y_hat_logits_list = []
        y_hat_logits_list.append(self.classifiers[0](z).squeeze())
        y_hat_logits_list.append(self.classifiers[1](z, alpha).squeeze())

        return y_hat_logits_list

    def compute_ce_losses(self, logits, labels):
        ce_losses = []
        for i, subspace_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.class_dims[i] == 1:
                subspace_logits = subspace_logits.squeeze()
                subspace_labels = subspace_labels.to(torch.float32)
                subspace_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    subspace_logits, subspace_labels
                )
            else:
                subspace_loss = torch.nn.functional.cross_entropy(
                    subspace_logits, subspace_labels
                )
            ce_losses.append(subspace_loss)
        return tuple(ce_losses)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss_cl, ce_loss_adv_cl = self.compute_ce_losses(logits, labels)
        overall_loss = ce_loss_cl + ce_loss_adv_cl
        return logits, ce_loss_cl, ce_loss_adv_cl, overall_loss
