from typing import Optional

import torch
import wandb
import numpy as np

from captcha_competition import MODELS_PATH
from captcha_competition.training import (
    CustomCategoricalCrossEntropyLoss,
    DataLoaderHandler,
    CustomAccuracyMetric,
)


INPUT_SHAPES = {
    "resnet": (1, 3, 64, 192),
    "efficientnet": (1, 3, 80, 210),
    "resnettransformer": (1, 3, 64, 192),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader_handler: DataLoaderHandler,
        val_dataloader_handler: DataLoaderHandler,
        epochs: int,
        verbose: bool = True,
        name: Optional[str] = None,
        num_samples: int = 10,
    ):
        self.model = model.to(DEVICE)
        self.model_type = model.__class__.__name__.lower()
        if verbose:
            from torchinfo import summary  # type: ignore

            print(f"Using device: {DEVICE}")
            summary(
                self.model,
                input_size=INPUT_SHAPES.get(self.model_type, (1, 3, 80, 200)),
            )

        self.optimizer = optimizer
        self.use_wandb = wandb.run is not None
        self.train_dataloader_handler = train_dataloader_handler
        self.val_dataloader_handler = val_dataloader_handler
        self.epochs = epochs
        self.accuracy_metric_per_digit = CustomAccuracyMetric()
        self.accuracy_metric = CustomAccuracyMetric(per_digit=False)
        self.verbose = verbose

        self.criterion = CustomCategoricalCrossEntropyLoss()

        if name is None and not self.use_wandb:
            raise ValueError("Name must be provided if not using wandb")
        self.name = name if name is not None else wandb.run.name  # type: ignore
        self.best_eval_accuracy = 0.0

        self.checkpoint_path = MODELS_PATH / f"{self.name}.pt"

        self.num_samples = num_samples

        if self.use_wandb:
            wandb.watch(self.model, criterion=self.criterion, log="all", log_freq=10)

    def train(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            losses = [0.0] * len(self.train_dataloader_handler)
            self.accuracy_metric.reset()
            for batch_idx, (images, labels) in enumerate(
                self.train_dataloader_handler, start=0
            ):
                loss = self.training_step(images, labels)
                losses[batch_idx] = loss
                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss}"
                    )
            # Log to wandb every epoch
            train_loss = sum(losses) / len(losses)
            train_accuracy = self.accuracy_metric.compute()
            train_accuracy_per_digit = self.accuracy_metric_per_digit.compute()
            eval_loss, eval_accuracy, eval_accuracy_per_digit, log_images = (
                self.evaluate()
            )
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss}, "
                    f"Train Accuracy: {train_accuracy}, "
                    f"Train Accuracy per digit: {train_accuracy_per_digit}, "
                    f"Eval Accuracy per digit: {eval_accuracy_per_digit}"
                    f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}, "
                )
            if self.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "eval_loss": eval_loss,
                        "eval_accuracy": eval_accuracy,
                        "train_accuracy_per_digit": train_accuracy_per_digit,
                        "eval_accuracy_per_digit": eval_accuracy_per_digit,
                        "incorrect_images": log_images,
                    }
                )
            if eval_accuracy > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_accuracy
                self.save_checkpoint(
                    epoch, eval_accuracy, eval_accuracy_per_digit
                )

    def evaluate(self):
        self.model.eval()
        losses = []
        self.accuracy_metric.reset()

        incorrect_images = []
        incorrect_preds = []
        incorrect_labels = []

        with torch.no_grad():
            for images, labels in self.val_dataloader_handler:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                self.accuracy_metric.update(outputs, labels)

                if len(incorrect_images) < self.num_samples and self.use_wandb:
                    self._update_incorrect_images(
                        incorrect_images,
                        incorrect_preds,
                        incorrect_labels,
                        images,
                        outputs,
                        labels,
                    )
        log_images = []
        if incorrect_images:
            log_images = self._crate_log_images(
                incorrect_images, incorrect_preds, incorrect_labels
            )

        return (
            sum(losses) / len(losses),
            self.accuracy_metric.compute(),
            self.accuracy_metric_per_digit.compute(),
            log_images,
        )

    def save_checkpoint(
        self, epoch: int, val_accuracy: float, val_accuracy_per_digit: float
    ):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "val_accuracy_per_digit": val_accuracy_per_digit,
            },
            self.checkpoint_path,
        )
        if self.verbose:
            print(f"Checkpoint saved: {self.checkpoint_path}")

    def training_step(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad()
        outputs: torch.Tensor = self.model(images)
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.accuracy_metric.update(outputs, labels)
        self.accuracy_metric_per_digit.update(outputs, labels)
        return loss.item()

    def _update_incorrect_images(
        self,
        incorrect_images: list[torch.Tensor],
        incorrect_preds: list[torch.Tensor],
        incorrect_labels: list[torch.Tensor],
        images: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        preds = CustomAccuracyMetric.get_predicted_classes(outputs)
        incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
        print(f"{incorrect_indices.shape =}")
        if len(incorrect_indices) > 0:
            for idx in incorrect_indices:
                if len(incorrect_images) >= self.num_samples:
                    break
                incorrect_images.append(images[idx].cpu())
                incorrect_preds.append(preds[idx].cpu())
                incorrect_labels.append(labels[idx].cpu())

    def _crate_log_images(
        self, incorrect_images, incorrect_preds, incorrect_labels
    ) -> list[wandb.Image]:
        log_images = []
        for img, pred, label in zip(
            incorrect_images, incorrect_preds, incorrect_labels
        ):
            img = img.permute(1, 2, 0)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            caption = f"Prediction: {pred.tolist()}, Label: {label.tolist()}"
            log_images.append(wandb.Image(img, caption=caption))
        return log_images
