"""
Classification training with per-epoch metric persistence.

After every epoch, metrics are written to:
    outputs/results/<run_id>_epochs.csv
    outputs/results/<run_id>_epochs.json
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from .base import BaseTask


class ClassificationTask(BaseTask):
    def __init__(self, run_id, config, newrun=True):
        self.task = "classification"
        super().__init__(run_id, config, newrun)
        self.history = []

    def _base_loss(self, logits, labels):
        """Compute the main classification loss without auxiliary losses."""
        if logits.ndim == 1 or (
            logits.ndim == 2 and logits.shape[-1] == 1
        ):
            return self.loss_fn(
                logits.reshape(-1),
                labels.to(logits.dtype).reshape(-1),
            )

        return self.loss_fn(logits, labels.long())

    def _probabilities(self, logits):
        """Convert model outputs into class probabilities."""
        logits = logits.float()

        if logits.ndim == 1 or (
            logits.ndim == 2 and logits.shape[-1] == 1
        ):
            positive_probability = torch.sigmoid(
                logits.reshape(-1)
            ).numpy()

            return np.column_stack(
                [
                    1.0 - positive_probability,
                    positive_probability,
                ]
            )

        return torch.softmax(logits, dim=-1).numpy()

    def _metrics(self, logits, targets, prefix):
        """Calculate classification metrics from logits and targets."""
        if isinstance(targets, torch.Tensor):
            targets = targets.int().numpy()
        else:
            targets = np.asarray(targets, dtype=np.int64)

        probabilities = self._probabilities(logits)
        predictions = probabilities.argmax(axis=1)

        scores = {
            f"{prefix}/accuracy": float(
                accuracy_score(targets, predictions)
            ),
            f"{prefix}/balanced_accuracy": float(
                balanced_accuracy_score(targets, predictions)
            ),
            f"{prefix}/f1_macro": float(
                f1_score(
                    targets,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            f"{prefix}/f1_weighted": float(
                f1_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            f"{prefix}/precision_macro": float(
                precision_score(
                    targets,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            f"{prefix}/recall_macro": float(
                recall_score(
                    targets,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

        try:
            if probabilities.shape[1] == 2:
                auroc = roc_auc_score(
                    targets,
                    probabilities[:, 1],
                )
            else:
                auroc = roc_auc_score(
                    targets,
                    probabilities,
                    labels=np.arange(probabilities.shape[1]),
                    multi_class="ovr",
                    average="macro",
                )

            scores[f"{prefix}/auroc_macro_ovr"] = float(auroc)

        except ValueError as error:
            print(f"[warning] Could not calculate {prefix} AUROC: {error}")
            scores[f"{prefix}/auroc_macro_ovr"] = float("nan")

        try:
            if probabilities.shape[1] == 2:
                auprc = average_precision_score(
                    targets,
                    probabilities[:, 1],
                )
            else:
                targets_one_hot = label_binarize(
                    targets,
                    classes=np.arange(probabilities.shape[1]),
                )

                auprc = average_precision_score(
                    targets_one_hot,
                    probabilities,
                    average="macro",
                )

            scores[f"{prefix}/auprc_macro"] = float(auprc)

        except ValueError as error:
            print(f"[warning] Could not calculate {prefix} AUPRC: {error}")
            scores[f"{prefix}/auprc_macro"] = float("nan")

        return scores

    def _evaluate(self, dataloader, prefix):
        """Evaluate one split and return its loss and metrics."""
        self.model.eval()

        all_logits = []
        all_targets = []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs in tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"Evaluating {prefix}",
            ):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.mixed,
                ):
                    logits = self.model(inputs)
                    labels = inputs["labels"].long()
                    loss = self._base_loss(logits, labels)

                batch_size = labels.shape[0]

                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

                all_logits.append(logits.float().detach().cpu())
                all_targets.append(labels.detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        scores = {
            f"{prefix}/loss": total_loss / max(total_samples, 1),
            **self._metrics(logits, targets, prefix),
        }

        self.log_scores(scores)
        return scores

    def _save_epoch_history(self):
        """Save all completed epochs immediately to JSON and CSV."""
        results_dir = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        json_path = results_dir / f"{self.run_id}_epochs.json"
        csv_path = results_dir / f"{self.run_id}_epochs.csv"

        json_record = {
            "run_id": self.run_id,
            "task": self.task,
            "model": self.config.model,
            "dataset": self.config.data.dataset,
            "completed_epochs": len(self.history),
            "epochs": self.history,
        }

        # Atomic JSON write.
        json_temporary = Path(str(json_path) + ".tmp")

        with open(json_temporary, "w") as file:
            json.dump(json_record, file, indent=2)

        json_temporary.replace(json_path)

        # Collect every metric name used in the history.
        fieldnames = []

        for record in self.history:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        # Atomic CSV write.
        csv_temporary = Path(str(csv_path) + ".tmp")

        with open(csv_temporary, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        csv_temporary.replace(csv_path)

        print(f"Saved epoch metrics: {csv_path}")
        print(f"Saved epoch history: {json_path}")

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(
                f"Epoch {epoch + 1}/"
                f"{self.config.training.epochs}"
            )

            self.model.train()

            train_logits = []
            train_targets = []

            total_training_loss = 0.0
            total_classification_loss = 0.0
            total_samples = 0

            for inputs in tqdm(
                self.train_dataloader,
                desc="Training",
            ):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.mixed,
                ):
                    logits = self.model(inputs)
                    labels = inputs["labels"].long()

                    classification_loss = self._base_loss(
                        logits,
                        labels,
                    )

                    loss = classification_loss

                    auxiliary_loss = getattr(
                        self.model,
                        "aux_loss",
                        None,
                    )

                    if auxiliary_loss is not None:
                        loss = loss + auxiliary_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_size = labels.shape[0]

                total_training_loss += (
                    float(loss.item()) * batch_size
                )
                total_classification_loss += (
                    float(classification_loss.item()) * batch_size
                )
                total_samples += batch_size

                self.log_step(loss.item())

                train_logits.append(
                    logits.float().detach().cpu()
                )
                train_targets.append(
                    labels.detach().cpu()
                )

            epoch_train_logits = torch.cat(
                train_logits,
                dim=0,
            )
            epoch_train_targets = torch.cat(
                train_targets,
                dim=0,
            )

            train_scores = {
                "train/loss": (
                    total_training_loss
                    / max(total_samples, 1)
                ),
                "train/classification_loss": (
                    total_classification_loss
                    / max(total_samples, 1)
                ),
                **self._metrics(
                    epoch_train_logits,
                    epoch_train_targets,
                    "train",
                ),
            }

            val_scores = self.val()
            test_scores = self.test()

            epoch_record = {
                "epoch": epoch + 1,
                **train_scores,
                **val_scores,
                **test_scores,
            }

            self.history.append(epoch_record)

            # Save immediately after every completed epoch.
            self._save_epoch_history()

            # Log metrics and save latest/best checkpoints.
            self.log_epoch(
                {
                    **train_scores,
                    **val_scores,
                    **test_scores,
                }
            )

            self.scheduler.step()

        self.model.eval()

    def val(self):
        return self._evaluate(
            self.val_dataloader,
            "val",
        )

    def test(self):
        return self._evaluate(
            self.test_dataloader,
            "test",
        )

    def predict(self, dataloader):
        """Return logits and targets for compatibility with test.py."""
        self.model.eval()

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for inputs in tqdm(
                dataloader,
                total=len(dataloader),
            ):
                inputs = self.prepare_batch(inputs)
                logits = self.model(inputs)

                all_logits.append(
                    logits.float().detach().cpu()
                )
                all_targets.append(
                    inputs["labels"].detach().cpu()
                )

        return (
            torch.cat(all_logits, dim=0),
            torch.cat(all_targets, dim=0),
        )

    def score(self, pred_scores, target):
        metrics = self._metrics(
            pred_scores,
            target,
            "temporary",
        )

        return {
            key.split("/", 1)[1]: value
            for key, value in metrics.items()
        }

    def build_loss(self):
        is_binary = self.train_dataset.n_classes == 2
        loss_name = self.config.training.loss

        if loss_name == "bce" or is_binary:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        elif loss_name in ("ce", "cross_entropy", "auto"):
            self.loss_fn = torch.nn.CrossEntropyLoss()

        else:
            raise ValueError(
                f"Invalid loss function selection: {loss_name}"
            )

        return self.loss_fn
