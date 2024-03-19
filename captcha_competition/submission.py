from captcha_competition import MODELS_PATH
from captha_competition.pytorch_model import EfficientNet, ResNet
from captcha_competition.data.preprocessing import image_to_tensor
import torch
from pathlib import Path
from captcha_competition import DATA_PATH
import os
import cv2
import numpy as np
import pandas as pd


def load_images():
    ids = [f.split(".")[0] for f in os.listdir(DATA_PATH / "kaggle")]
    filepaths = [DATA_PATH / "kaggle" / f"{f:05}.png" for f in ids]
    images = np.array([cv2.imread(str(f)) for f in filepaths])
    # Convert to tensor
    images = np.array([image_to_tensor(img) for img in images])
    return images, ids


def load_model(model_path, parameters):
    model = None
    if "efficientnet" in model_path:
        model = EfficientNet(**parameters)
    elif "resnet" in model_path:
        model = ResNet(**parameters)
    else:
        raise ValueError("Model not found")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def submit(model: str, parameters: dict):
    model_path = MODELS_PATH / model
    model = load_model(model_path, parameters)
    images, ids = load_images()
    output = model(images)
    preds = output.argmax(dim=1)
    df = pd.DataFrame({"Id": ids, "Label": preds})
    df.to_csv(DATA_PATH / "submission.csv", index=False)
