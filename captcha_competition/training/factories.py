from typing import Optional, Callable
from pathlib import Path

import torch
import numpy as np

from captcha_competition.pytorch_model import ResNet, EfficientNet
from captcha_competition.data import (
    CaptchaDataset,
    SyntheticCaptchaIterableDataset,
)
from captcha_competition.data.preprocessing_pipelines import (
    create_preprocessing_pipeline,
)
from captcha_competition.training import DataLoaderHandler, Trainer
from captcha_competition import DATA_RAW_PATH


def trainer_factory(
    model_params: dict,
    optimizer_params: dict,
    preprocessing_params: dict,
    train_dataset_params: dict,
    val_dataset_params: dict,
    dataloader_params: dict,
    trainer_params: dict,
):
    model = model_factory(**model_params)
    optimizer = optimizer_factory(model, **optimizer_params)

    preprocessing_fc = preprocessing_fc_factory(
        model_type=model_params["model_type"], **preprocessing_params
    )
    dataset = dataset_factory(
        preprocessing_fc=preprocessing_fc, **train_dataset_params
    )
    dataloader = DataLoaderHandler(dataset, **dataloader_params)
    test_dataset = dataset_factory(
        preprocessing_fc=preprocessing_fc, **val_dataset_params
    )
    val_dataloader = DataLoaderHandler(test_dataset, **dataloader_params)
    return Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader_handler=dataloader,
        val_dataloader_handler=val_dataloader,
        name=get_model_name(model_params, train_dataset_params),
        **trainer_params,
    )


def get_model_name(model_params: dict, dataset_params: dict) -> str:
    return (
        f"{model_params['model_type']}_"
        f"{model_params['initial_filters']}_"
        f"{model_params['multiplier']}_"
        f"{dataset_params['dataset_type']}"
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
    preprocessing_fc: Callable[[np.ndarray], np.ndarray],
    folder_name: str = "train",
    data_path: Path = DATA_RAW_PATH,
):
    if dataset_type == "real":
        return CaptchaDataset(
            raw_data_dir=data_path,
            folder_name=folder_name,
            preprocessing_fc=preprocessing_fc,
        )
    if dataset_type == "synthetic":
        return SyntheticCaptchaIterableDataset(
            preprocessing_fc=preprocessing_fc,
        )
    raise ValueError(f"Dataset type {dataset_type} not supported")


def preprocessing_fc_factory(
    model_type: str = "",
    use_full_preprocessing: bool = False,
    preprocessing_steps: Optional[list[str]] = None,
) -> Callable[[np.ndarray], np.ndarray]:

    if preprocessing_steps is not None:
        return create_preprocessing_pipeline(preprocessing_steps)

    if use_full_preprocessing:
        preprocessing_steps = ["remove_background", "to_grayscale", "closing"]
    else:
        preprocessing_steps = []

    if model_type == "resnet":
        preprocessing_steps.append("resize_resnet")
    elif model_type == "efficientnet":
        preprocessing_steps.append("resize_efficientnet")
    else:
        raise ValueError(f"Model type {model_type} not supported")

    return create_preprocessing_pipeline(preprocessing_steps)
