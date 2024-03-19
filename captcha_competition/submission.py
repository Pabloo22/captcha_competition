from typing import Any  

import torch
from pathlib import Path
from captcha_competition import DATA_PATH
import os
import cv2
import numpy as np
import pandas as pd

from captcha_competition import MODELS_PATH
from captcha_competition.pytorch_model import EfficientNet, ResNet
from captcha_competition.data.preprocessing_pipelines import image_to_tensor


def load_images():
    ids = [f.split(".")[0] for f in os.listdir(DATA_PATH / "kaggle")]
    filepaths = [DATA_PATH / "kaggle" / f"{f:05}.png" for f in ids]
    images = np.array([cv2.imread(str(f)) for f in filepaths])
    # Convert to tensor
    images = np.array([image_to_tensor(img) for img in images])
    return images, ids


def load_model(model_path: Path, parameters: dict[str, Any]):
    if "efficientnet" in str(model_path):
        model: torch.nn.Module = EfficientNet(**parameters)
    elif "resnet" in str(model_path):
        model: torch.nn.Module = ResNet(**parameters)  # type: ignore
    else:
        raise ValueError("Model not found")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def submit(filename: str, parameters: dict):
    model_path = MODELS_PATH / filename
    model = load_model(model_path, parameters)
    images, ids = load_images()
    images = torch.tensor(images).float() / 255
    output = model(images)
    preds = output.argmax(dim=1)
    df = pd.DataFrame({"Id": ids, "Label": preds})
    df.to_csv(DATA_PATH / "submission.csv", index=False)


if __name__ == "__main__":
    pass
