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


INPUT_SHAPES = {"resnet": (3, 64, 192), "efficientnet": (3, 80, 210)}

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
        if verbose:
            from torchsummary import summary  # type: ignore

            print(f"Using device: {DEVICE}")
            model_type = model.__class__.__name__.lower()
            summary(model, input_size=INPUT_SHAPES[model_type])

        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.use_wandb = wandb.run is not None
        self.train_dataloader_handler = train_dataloader_handler
        self.val_dataloader_handler = val_dataloader_handler
        self.epochs = epochs
        self.accuracy_metric = CustomAccuracyMetric()
        self.verbose = verbose

        self.criterion = CustomCategoricalCrossEntropyLoss()

        if name is None and not self.use_wandb:
            raise ValueError("Name must be provided if not using wandb")
        self.name = name if name is not None else wandb.run.name  # type: ignore
        self.best_eval_accuracy = 0.0

        self.checkpoint_path = MODELS_PATH / f"{self.name}.pt"

        self.num_samples = num_samples

        if self.use_wandb:
            # Watch the model
            wandb.watch(self.model, criterion=self.criterion, log="all")

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
            eval_loss, eval_accuracy, log_images = self.evaluate()
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss}, "
                    f"Train Accuracy: {train_accuracy}, "
                    f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}"
                )
            if self.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "eval_loss": eval_loss,
                        "eval_accuracy": eval_accuracy,
                        "incorrect_images": log_images,
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
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.accuracy_metric.update(outputs, labels)
        return loss.item()

    def evaluate(self):
        self.model.eval()
        losses = []
        self.accuracy_metric.reset()

        incorrect_images = []
        incorrect_preds = []
        incorrect_labels = []

        with torch.no_grad():
            for images, labels in self.val_dataloader_handler:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
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

        # Log incorrect samples to wandb
        if incorrect_images:
            # Log incorrect samples to wandb
            log_images = []
            for img, pred, label in zip(
                incorrect_images, incorrect_preds, incorrect_labels
            ):
                img = img.permute(1, 2, 0)  # Convert CHW to HWC
                img = img.numpy()  # Convert tensor to numpy array
                img = (img * 255).astype(np.uint8)
                # Now img is ready for conversion to PIL Image within wandb.Image
                caption = f"Pred: {pred.tolist()}, Label: {label.tolist()}"
                log_images.append(wandb.Image(img, caption=caption))

        return (
            sum(losses) / len(losses),
            self.accuracy_metric.compute(),
            log_images,
        )

    def _update_incorrect_images(
        self,
        incorrect_images: list[torch.Tensor],
        incorrect_preds: list[torch.Tensor],
        incorrect_labels: list[torch.Tensor],
        images: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        preds = outputs.argmax(dim=1)
        incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
        print(f"{incorrect_indices.shape =}")
        if len(incorrect_indices) > 0:
            for idx in incorrect_indices:
                if len(incorrect_images) >= self.num_samples:
                    break
                incorrect_images.append(images[idx].cpu())
                incorrect_preds.append(preds[idx].cpu())
                incorrect_labels.append(labels[idx].cpu())
