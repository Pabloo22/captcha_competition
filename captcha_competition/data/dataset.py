from typing import Callable
from pathlib import Path
import os

import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image  # type: ignore
from torch.utils.data import Dataset
import imageio

from captcha_competition import DATA_RAW_PATH, DATA_PROCESSED_PATH
from captcha_competition.data.preprocessing_pipelines import (
    label_to_tensor,
    image_to_tensor,
)

MAPPING = {
    "a": 10,
    "e": 11,
    "u": 12,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptchaDataset(Dataset):
    def __init__(
        self,
        folder_name: str = "train",
        preprocessing_fc: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        raw_data_dir: Path = DATA_RAW_PATH,
        processed_data_dir: Path = DATA_PROCESSED_PATH,
        remove_previously_processed: bool = False,
        save_processed: bool = False,
        zero_pad: int = 5,
        only_tensors: bool = False,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.raw_img_dir = os.path.join(raw_data_dir, folder_name)
        self.zero_pad = zero_pad
        self.only_tensors = only_tensors
        self.pin_memory = pin_memory

        self.processed_data_dir = processed_data_dir
        self.processed_img_dir = os.path.join(processed_data_dir, folder_name)

        os.makedirs(self.processed_img_dir, exist_ok=True)

        if remove_previously_processed:
            for img_name in os.listdir(self.processed_img_dir):
                os.remove(os.path.join(self.processed_img_dir, img_name))

        labels_path = os.path.join(raw_data_dir, f"{folder_name}.csv")
        # Check if exists
        if os.path.exists(labels_path):
            self.img_labels = pd.read_csv(labels_path)
        else:
            self.img_labels = None  # type: ignore[assignment]

        self.preprocessing_fc = preprocessing_fc
        self.save_processed = save_processed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = (
            str(self.img_labels.iloc[idx, 0]).zfill(self.zero_pad) + ".png"
        )
        raw_img_path = os.path.join(self.raw_img_dir, img_name)
        processed_img_path = os.path.join(self.processed_img_dir, img_name)

        if not os.path.exists(processed_img_path):
            if not self.only_tensors:
                raw_image = imageio.imread(raw_img_path)
                processed_image = self.preprocessing_fc(raw_image)
                if self.save_processed:
                    save_img = (processed_image * 255).astype(np.uint8)
                    imageio.imwrite(processed_img_path, save_img)
                image_tensor = image_to_tensor(processed_image)

            # If True, the preprocessing function receives and returns tensors
            if self.only_tensors:
                image_tensor = self._read_image_with_torch(raw_img_path)
                image_tensor = self.preprocessing_fc(image_tensor)
        else:
            image_tensor = self._read_image_with_torch(processed_img_path)

        label_tensor = self._get_label_tensor(idx)
        return image_tensor, label_tensor

    def _read_image_with_torch(self, img_path) -> torch.Tensor:
        image_tensor = read_image(img_path)
        image_tensor = image_tensor.float() / 255.0
        # Raises error if pin_memory is True in the DataLoader:
        if not self.pin_memory:
            image_tensor = image_tensor.to(DEVICE)
        return image_tensor

    @staticmethod
    def _to_int(label: str) -> int:
        if label.isdigit():
            return int(label)
        return MAPPING[label]

    def _get_label_tensor(self, idx):
        if self.img_labels is None:
            return None
        label = self.img_labels.iloc[idx, 1]

        label = [self._to_int(i) for i in str(label).zfill(self.zero_pad)]
        return label_to_tensor(label)


if __name__ == "__main__":
    pass
