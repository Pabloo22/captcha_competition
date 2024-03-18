from typing import Callable
from pathlib import Path
import os

import numpy as np
import pandas as pd
from torchvision.io import read_image  # type: ignore
from torch.utils.data import Dataset
import imageio

from captcha_competition import DATA_RAW_PATH, DATA_PROCESSED_PATH
from captcha_competition.data import label_to_tensor, image_to_tensor


class CaptchaDataset(Dataset):
    def __init__(
        self,
        folder_name: str = "train",
        preprocessing_fc: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        raw_data_dir: Path = DATA_RAW_PATH,
        processed_data_dir: Path = DATA_PROCESSED_PATH,
        remove_previously_processed: bool = False,
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

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(self.img_labels.iloc[idx, 0]).zfill(5) + ".png"
        raw_img_path = os.path.join(self.raw_img_dir, img_name)
        processed_img_path = os.path.join(self.processed_img_dir, img_name)

        if not os.path.exists(processed_img_path):
            raw_image = imageio.imread(raw_img_path)
            processed_image = self.preprocessing_fc(raw_image)
            imageio.imwrite(processed_img_path, processed_image)
            image_tensor = image_to_tensor(processed_image)
        else:
            image_tensor = read_image(processed_img_path)

        label = self.img_labels.iloc[idx, 1]
        label = [int(i) for i in str(label)]

        label_tensor = label_to_tensor(label)
        return image_tensor, label_tensor


if __name__ == "__main__":
    pass