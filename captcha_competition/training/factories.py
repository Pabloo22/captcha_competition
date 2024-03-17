from typing import Optional, Callable

from pathlib import Path

import torch
import numpy as np

from captcha_competition.pytorch_model import ResNet, EfficientNet
from captcha_competition.data import (
    CaptchaDataset,
    SyntheticCaptchaIterableDataset,
    remove_bg_v1,
    image_to_tensor,
)
from captcha_competition.training import DataLoaderHandler, Trainer
from captcha_competition import DATA_RAW_PATH


def trainer_factory(
    model_params: dict,
    optimizer_params: dict,
    dataset_params: dict,
    dataloader_params: dict,
    trainer_params: dict,
):
    model = model_factory(**model_params)
    optimizer = optimizer_factory(model, **optimizer_params)
    dataset = dataset_factory(**dataset_params)
    dataloader = DataLoaderHandler(dataset, **dataloader_params)
    return Trainer(
        model=model,
        optimizer=optimizer,
        data_loader_handler=dataloader,
        **trainer_params,
    )


def model_factory(model_type: str, initial_filters: int, multiplier: float):
    if model_type == "resnet":
        return ResNet(initial_filters, multiplier)
    if model_type == "efficientnet":
        return EfficientNet(initial_filters, multiplier)
    raise ValueError(f"Model type {model_type} not supported")


def optimizer_factory(
    model: torch.nn.Module, learning_rate: float, weight_decay: float = 0.0
):
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def dataset_factory(
    dataset_type: str,
    folder_name: str = "train",
    preprocessing_fc_name: Optional[str] = None,
    data_path: Path = DATA_RAW_PATH,
):
    preprocessing_fc = preprocessing_fc_factory(preprocessing_fc_name)
    if dataset_type == "real":
        return CaptchaDataset(
            data_dir=data_path,
            folder_name=folder_name,
            transform=preprocessing_fc,
        )
    if dataset_type == "synthetic":
        return SyntheticCaptchaIterableDataset(
            preprocessing_fc=preprocessing_fc,
        )
    raise ValueError(f"Dataset type {dataset_type} not supported")


def preprocessing_fc_factory(
    preprocessing_fc: Optional[str],
) -> Callable[[np.ndarray], torch.Tensor]:
    if preprocessing_fc == "remove_bg_v1":
        return remove_bg_v1
    if preprocessing_fc is None:
        return image_to_tensor
    raise ValueError(
        f"Preprocessing function {preprocessing_fc} not supported"
    )
