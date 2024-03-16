import torch
import wandb

from captcha_competition.training import (
    CustomCategoricalCrossEntropyLoss,
    DataLoaderHandler,
    CustomAccuracyMetric,
)


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader_handler: DataLoaderHandler,
        epochs: int,
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_wandb = wandb.run is not None
        self.data_loader_handler = data_loader_handler
        self.epochs = epochs
        self.accuracy_metric = CustomAccuracyMetric()
        self.verbose = verbose

    def train(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            losses = [0.0] * len(self.data_loader_handler)
            self.accuracy_metric.reset()
            for batch_idx, (images, labels) in enumerate(
                self.data_loader_handler, start=1
            ):
                loss = self.training_step(images, labels)
                losses[batch_idx] = loss
                if batch_idx % 100 == 0 and self.verbose:
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss}"
                    )
            # Log to wandb every epoch
            if self.use_wandb:
                wandb.log(
                    {
                        "loss": sum(losses) / len(losses),
                        "acc": self.accuracy_metric.compute(),
                    }
                )

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
