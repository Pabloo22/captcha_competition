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
    ):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.raw_img_dir = os.path.join(raw_data_dir, folder_name)

        self.processed_data_dir = processed_data_dir
        self.processed_img_dir = os.path.join(processed_data_dir, folder_name)

        os.makedirs(self.processed_img_dir, exist_ok=True)

        if remove_previously_processed:
            for img_name in os.listdir(self.processed_img_dir):
                os.remove(os.path.join(self.processed_img_dir, img_name))

        self.img_labels = pd.read_csv(
            os.path.join(raw_data_dir, f"{folder_name}.csv")
        )

        self.preprocessing_fc = preprocessing_fc
        self.save_processed = save_processed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(self.img_labels.iloc[idx, 0]).zfill(5) + ".png"
        raw_img_path = os.path.join(self.raw_img_dir, img_name)
        processed_img_path = os.path.join(self.processed_img_dir, img_name)

        if not os.path.exists(processed_img_path):
            raw_image = imageio.imread(raw_img_path)
            processed_image = self.preprocessing_fc(raw_image)
            if self.save_processed:
                save_img = (processed_image * 255).astype(np.uint8)
                imageio.imwrite(processed_img_path, save_img)
            image_tensor = image_to_tensor(processed_image)
        else:
            image_tensor = self._read_image_with_torch(processed_img_path)

        label_tensor = self._get_label_tensor(idx)
        return image_tensor, label_tensor

    @staticmethod
    def _read_image_with_torch(img_path) -> torch.Tensor:
        image_tensor = read_image(img_path)
        image_tensor = image_tensor.float() / 255.0
        # Raises error if pin_memory is True in the DataLoader:
        # image_tensor = image_tensor.to(DEVICE)
        return image_tensor

    def _get_label_tensor(self, idx):
        label = self.img_labels.iloc[idx, 1]
        label = [int(i) for i in str(label).zfill(6)]
        return label_to_tensor(label)


if __name__ == "__main__":
    pass
