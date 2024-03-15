import torch
from torch.utils.data import DataLoader
import wandb


from captcha_competition.training import CustomCategoricalCrossEntropyLoss, DataLoaderHandler


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader_handler: DataLoaderHandler,
        wandb_project=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.wandb_project = wandb_project
        self.data_loader_handler = data_loader_handler

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (images, labels) in enumerate(self.data_loader_handler):
                loss = self.training_step(images, labels)

    def training_step(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss_fn = CustomCategoricalCrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
