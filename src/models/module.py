from typing import List, Optional

import lightning.pytorch as pl
import torch
import torchmetrics

from src.metrics.metrics import SimpleMetric
from src.models.method_abstraction import SubspaceDistanglementMethod


class SubspaceEncoderModule(pl.LightningModule):
    """Image encoder module for lightning training.

    Attributes:
        method: Optimization method.
        encoder: Encoder backbone architecture.
        class_dims: List of class dimensions that are encoded into subspaces.
        warmup_epochs: Number of warmup epochs.
        monitor_metric: Which metric to monitor for best model.
        save_after_x_epochs: Only save models after x epochs.
        save_only_last_epoch: Only save model of last epoch.
        n_test: Number of test sets.
    """

    def __init__(
        self,
        method: SubspaceDistanglementMethod,
        encoder: torch.nn.Module,
        class_dims: List[int] = [8, 20],
        warmup_epochs: int = 0,
        monitor_metric: str = "valid_loss",
        save_after_x_epochs: Optional[int] = None,
        save_only_last_epoch: bool = False,
        n_test: int = 2,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = encoder
        self.method = method
        self.warmup_epochs = warmup_epochs
        self.n_test = n_test
        self.monitor_metric = monitor_metric
        self.save_after_x_epochs = save_after_x_epochs
        self.save_only_last_epoch = save_only_last_epoch

        num_classifiers = len(class_dims)
        self.c_dims = class_dims
        self.warmup_epochs = warmup_epochs
        self.monitor_metric = monitor_metric
        self.n_test = n_test

        # Initialize metrics for epoch-wise logging.
        self.train_subspace_cs_loss = SimpleMetric()
        self.train_dmeasure_z1_z2 = SimpleMetric()
        self.train_loss = SimpleMetric()
        self.p_value = SimpleMetric()

        self.val_subspace_cs_loss = SimpleMetric()
        self.val_dmeasure_z1_z2 = SimpleMetric()
        self.val_loss = SimpleMetric()

        cs_metrics = self._subspace_classification_metrics(
            num_classifiers, state="train"
        )
        val_cs_metrics = self._subspace_classification_metrics(
            num_classifiers, state="val"
        )

        test_subspace_cs_loss = []
        test_dmeasure_z1_z2 = []
        test_loss = []
        test_cs_metrics = []
        for dataloader_idx in range(n_test):
            test_subspace_cs_loss.append(SimpleMetric())
            test_dmeasure_z1_z2.append(SimpleMetric())
            test_loss.append(SimpleMetric())
            metrics = self._subspace_classification_metrics(
                num_classifiers, state="test", data_index=dataloader_idx
            )
            for metric in metrics:
                test_cs_metrics.append(metric)

        self.cs_metrics = torch.nn.ModuleList(cs_metrics)
        self.val_cs_metrics = torch.nn.ModuleList(val_cs_metrics)
        self.best_val_loss = 1000
        self.best_val_c0_acc = 0

        self.test_cs_metrics = torch.nn.ModuleList(test_cs_metrics)
        self.test_subspace_cs_loss = torch.nn.ModuleList(test_subspace_cs_loss)
        self.test_dmeasure_z1_z2 = torch.nn.ModuleList(test_dmeasure_z1_z2)
        self.test_loss = torch.nn.ModuleList(test_loss)

    def _subspace_classification_metrics(
        self, num_classifiers, state: str = "train", data_index: int = None
    ):
        metrics_list = []
        for i in range(num_classifiers):
            if self.c_dims[i] == 1:
                metrics = torchmetrics.MetricCollection(
                    [
                        torchmetrics.classification.BinaryAccuracy(),
                        torchmetrics.classification.BinaryJaccardIndex(),
                        torchmetrics.classification.BinaryAUROC(),
                    ]
                )
            else:
                num_classes = self.c_dims[i]
                metrics = torchmetrics.MetricCollection(
                    [
                        torchmetrics.classification.MulticlassAccuracy(
                            num_classes, average="micro"
                        ),
                        torchmetrics.classification.MulticlassJaccardIndex(
                            num_classes, average="macro"
                        ),
                        torchmetrics.classification.MulticlassAUROC(
                            num_classes, average="macro"
                        ),
                    ]
                )
            if data_index is not None:
                prefix = f"{state}{data_index}_c{i}_"
            else:
                prefix = f"{state}_c{i}_"
            metrics_list.append(metrics.clone(prefix=prefix))
        return metrics_list

    def _class_metric_logging(
        self,
        logits,
        labels,
        state: str = "train",
        dataloader_idx: Optional[int] = None,
    ):
        for i, y_hat_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.c_dims[i] == 1:
                y_hat_logits = y_hat_logits.squeeze()
                subspace_labels = subspace_labels.to(torch.float32)
                y_hat = torch.nn.functional.sigmoid(y_hat_logits)
            else:
                y_hat = torch.nn.functional.log_softmax(y_hat_logits, dim=1)
            if state == "train":
                self.cs_metrics[i].update(y_hat, subspace_labels)
            elif state == "val":
                self.val_cs_metrics[i].update(y_hat, subspace_labels)
            elif state == "test":
                self.test_cs_metrics[i + len(self.c_dims) * dataloader_idx].update(
                    y_hat, subspace_labels
                )

    def configure_optimizers(self):
        return self.method.configure_optimizers(self.encoder)

    def forward(self, batch):
        image = batch[0]
        latent_space = self.encoder(image)
        return latent_space

    def training_step(self, batch, batch_idx):
        z = self.forward(batch)
        labels = batch[1]

        total_steps = self.trainer.estimated_stepping_batches
        start_steps = self.current_epoch * self.trainer.num_training_batches
        p = float(batch_idx + start_steps) / total_steps

        ce_loss, dependence_estimate, overall_loss, logits = self.method.gradient_step(
            lightning_module=self,
            z=z,
            labels=labels,
            batch_idx=batch_idx,
            warmup=self.current_epoch < self.warmup_epochs,
            p=p,
        )

        self.train_subspace_cs_loss.update(ce_loss.detach())
        self.train_dmeasure_z1_z2.update(dependence_estimate.detach())
        self.train_loss.update(overall_loss.detach())
        self.p_value.update(p)

        self._class_metric_logging(
            logits,
            labels,
            state="train",
        )

    def validation_step(self, batch, batch_idx):
        z = self.forward(batch)
        labels = batch[1]

        logits, ce_loss, dependence_estimate, overall_loss = self.method.get_losses(
            z,
            labels,
        )

        self.val_subspace_cs_loss.update(ce_loss.detach())
        self.val_dmeasure_z1_z2.update(dependence_estimate.detach())
        if self.save_after_x_epochs is not None:
            if self.current_epoch < self.save_after_x_epochs:
                if self.monitor_metric == "valid_loss":
                    overall_loss *= 10.0
        self.val_loss.update(overall_loss.detach())

        self._class_metric_logging(
            logits,
            labels,
            state="val",
        )

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        z = self.forward(batch)
        labels = batch[1]

        logits, ce_loss, dependence_estimate, overall_loss = self.method.get_losses(
            z, labels
        )

        self.test_subspace_cs_loss[dataloader_idx].update(ce_loss.detach())
        self.test_dmeasure_z1_z2[dataloader_idx].update(dependence_estimate.detach())
        self.test_loss[dataloader_idx].update(overall_loss.detach())

        self._class_metric_logging(
            logits,
            labels,
            state="test",
            dataloader_idx=dataloader_idx,
        )

    def on_train_epoch_end(self):
        metric_dict = {
            "train_dmeasure_z1_z2": self.train_dmeasure_z1_z2.compute(),
            "train_ce_loss": self.train_subspace_cs_loss.compute(),
            "train_loss": self.train_loss.compute(),
            "p": self.p_value.compute(),
            "step": float(self.current_epoch),
        }

        for i in range(len(self.c_dims)):
            metric_dict.update(self.cs_metrics[i].compute())

        self.log_dict(
            metric_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # manually reset metrics
        self.train_dmeasure_z1_z2.reset()
        self.train_subspace_cs_loss.reset()
        self.train_loss.reset()
        self.p_value.reset()
        for i in range(len(self.c_dims)):
            self.cs_metrics[i].reset()

    def _set_best_val_metrics(self, metric_dict):
        self.best_val_metrics = {
            "best_epoch": self.current_epoch,
            "best_c0_acc": (
                metric_dict["val_c0_BinaryAccuracy"]
                if self.c_dims[0] == 1
                else metric_dict["val_c0_MulticlassAccuracy"]
            ),
            "best_c0_auroc": (
                metric_dict["val_c0_BinaryAUROC"]
                if self.c_dims[0] == 1
                else metric_dict["val_c0_MulticlassAUROC"]
            ),
        }
        if len(self.c_dims) > 1:
            self.best_val_metrics["best_c1_acc"] = (
                metric_dict["val_c1_BinaryAccuracy"]
                if self.c_dims[0] == 1
                else metric_dict["val_c1_MulticlassAccuracy"]
            )
            self.best_val_metrics["best_c1_auroc"] = (
                metric_dict["val_c1_BinaryAUROC"]
                if self.c_dims[0] == 1
                else metric_dict["val_c1_MulticlassAUROC"]
            )
            self.best_val_metrics["best_dmeasure"] = metric_dict["val_dmeasure_z1_z2"]

    def on_validation_epoch_end(self):
        metric_dict = {
            "val_dmeasure_z1_z2": self.val_dmeasure_z1_z2.compute(),
            "val_ce_loss": self.val_subspace_cs_loss.compute(),
            "valid_loss": self.val_loss.compute(),
            "step": float(self.current_epoch),
        }

        for i in range(len(self.c_dims)):
            metric_dict.update(self.val_cs_metrics[i].compute())

        if self.current_epoch > 0:
            if self.save_only_last_epoch and (
                self.current_epoch == (self.trainer.max_epochs - 1)
            ):
                self._set_best_val_metrics(metric_dict)
                self.best_val_loss = metric_dict["valid_loss"]
            elif (self.monitor_metric == "valid_loss") and (
                metric_dict[self.monitor_metric] < self.best_val_loss
            ):
                self._set_best_val_metrics(metric_dict)
                self.best_val_loss = metric_dict["valid_loss"]
            elif "val_c0" in self.monitor_metric:
                if metric_dict[self.monitor_metric] > self.best_val_c0_acc:
                    self._set_best_val_metrics(metric_dict)
                    self.best_val_c0_acc = metric_dict[self.monitor_metric]

            self.log_dict(
                self.best_val_metrics,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self.log_dict(
            metric_dict,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # manually reset metrics
        self.val_dmeasure_z1_z2.reset()
        self.val_subspace_cs_loss.reset()
        self.val_loss.reset()
        for i in range(len(self.c_dims)):
            self.val_cs_metrics[i].reset()

    def on_test_epoch_end(self):
        metric_dict = {
            "step": float(self.current_epoch),
        }
        for dataloader_idx in range(self.n_test):
            test_dmeasure_z1_z2 = self.test_dmeasure_z1_z2[dataloader_idx].compute()
            test_subspace_cs_loss = self.test_subspace_cs_loss[dataloader_idx].compute()
            test_loss = self.test_loss[dataloader_idx].compute()
            log_dict = {
                f"test{dataloader_idx}_dmeasure_z1_z2": test_dmeasure_z1_z2,
                f"test{dataloader_idx}_ce_loss": test_subspace_cs_loss,
                f"test{dataloader_idx}_loss": test_loss,
            }
            metric_dict.update(log_dict)
            for i in range(len(self.c_dims)):
                metric_dict.update(
                    self.test_cs_metrics[
                        i + len(self.c_dims) * dataloader_idx
                    ].compute()
                )

        self.log_dict(
            metric_dict,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # manually reset metrics
        for dataloader_idx in range(self.n_test):
            self.test_dmeasure_z1_z2[dataloader_idx].reset()
            self.test_subspace_cs_loss[dataloader_idx].reset()
            self.test_loss[dataloader_idx].reset()
            for i in range(len(self.c_dims)):
                self.test_cs_metrics[i + len(self.c_dims) * dataloader_idx].reset()
