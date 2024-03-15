"""Returns the train and validation data from the data folder.

The structure of the data folder is as follows:
data
├── raw
│   ├── train
│   │   ├── (.png images)
│   ├── validation
│   │   ├── (.png images)
|   |── train.csv
|   |── validation.csv
├── processed
│   ├── train
│   │   ├── (.png images)
│   ├── validation
│   │   ├── (.png images)

Each image name is the id of the image. Example: 00000.png
Each csv file has two columns: id and label. First two rows of train.csv:
Id,Label
0,024706
"""
from typing import Iterable
from pathlib import Path

import pandas as pd
import numpy as np
import cv2

from captcha_competition import DATA_RAW_PATH


def load_raw_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_train, y_train, X_val, y_val as numpy arrays."""
    train_df = _load_csv_data(DATA_RAW_PATH / "train.csv")
    val_df = _load_csv_data(DATA_RAW_PATH / "validation.csv")

    X_train = _load_images(DATA_RAW_PATH / "train", train_df["Id"])
    X_val = _load_images(DATA_RAW_PATH / "validation", val_df["Id"])

    y_train = train_df["Label"].values
    y_val = val_df["Label"].values

    return X_train, y_train, X_val, y_val  # type: ignore


def _load_images(directory: Path, ids: Iterable) -> np.ndarray:
    """Loads images from the directory based on the given IDs."""
    return np.array([cv2.imread(str(directory / f"{i:05}.png")) for i in ids])


def _load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Loads CSV data from the given path."""
    return pd.read_csv(csv_path)
