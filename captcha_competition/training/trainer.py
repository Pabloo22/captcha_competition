from typing import Optional

import torch
import wandb

from captcha_competition import MODELS_PATH
from captcha_competition.training import (
    CustomCategoricalCrossEntropyLoss,
    DataLoaderHandler,
    CustomAccuracyMetric,
)


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
    ):
        if verbose:
            print(f"Using device: {DEVICE}")
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.use_wandb = wandb.run is not None
        self.train_dataloader_handler = train_dataloader_handler
        self.val_dataloader_handler = val_dataloader_handler
        self.epochs = epochs
        self.accuracy_metric = CustomAccuracyMetric()
        self.verbose = verbose

        if name is None and not self.use_wandb:
            raise ValueError("Name must be provided if not using wandb")
        self.name = name if name is not None else wandb.run.name  # type: ignore
        self.best_eval_accuracy = 0.0

        self.checkpoint_path = MODELS_PATH / f"{self.name}.pt"

    def train(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            losses = [0.0] * len(self.train_dataloader_handler)
            self.accuracy_metric.reset()
            for batch_idx, (images, labels) in enumerate(
                self.train_dataloader_handler, start=0
            ):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                loss = self.training_step(images, labels)
                losses[batch_idx] = loss
                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss}"
                    )
            # Log to wandb every epoch
            train_loss = sum(losses) / len(losses)
            train_accuracy = self.accuracy_metric.compute()
            eval_loss, eval_accuracy = self.evaluate()
            if self.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "eval_loss": eval_loss,
                        "eval_accuracy": eval_accuracy,
                    }
                )
            if eval_accuracy > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_accuracy
                self.save_checkpoint(epoch, eval_accuracy)

    def save_checkpoint(self, epoch: int, val_accuracy: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_accuracy": val_accuracy,
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
        loss_fn = CustomCategoricalCrossEntropyLoss()
        loss: torch.Tensor = loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.accuracy_metric.update(outputs, labels)
        return loss.item()

    def evaluate(self):
        self.model.eval()
        losses = []
        self.accuracy_metric.reset()
        with torch.no_grad():
            for images, labels in self.val_dataloader_handler:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss_fn = CustomCategoricalCrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                losses.append(loss.item())
                self.accuracy_metric.update(outputs, labels)
        return sum(losses) / len(losses), self.accuracy_metric.compute()
