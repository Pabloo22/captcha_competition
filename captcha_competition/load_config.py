from typing import Any
from pathlib import Path
from enum import Enum

import yaml

from captcha_competition import CONFIG_PATH


class ConfigKeys(str, Enum):
    TRAINER = "trainer"
    MODEL = "model"
    OPTIMIZER = "optimizer"
    PREPROCESSING = "preprocessing"
    TRAIN_DATASET = "train_dataset"
    VAL_DATASET = "val_dataset"
    DATALOADER = "dataloader"


def load_config(
    filename: str, config_dir: Path = CONFIG_PATH
) -> dict[str, dict[str, Any]]:
    with open(config_dir / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config("resnet_default.yaml")
    print(config)
    print(config[ConfigKeys.MODEL])
    print(config[ConfigKeys.MODEL]["model_type"])

    print(config[ConfigKeys.TRAIN_DATASET])
    print(config[ConfigKeys.TRAIN_DATASET]["dataset_type"])
