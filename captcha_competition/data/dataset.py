import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from captcha_competition import DATA_RAW_PATH
from generate_captcha import label_to_tensor, image_to_tensor


class CaptchaDataset(Dataset):
    def __init__(
        self,
        data_dir=DATA_RAW_PATH,
        folder_name="train",
        transform=image_to_tensor,
        target_transform=label_to_tensor,
    ):
        self.data_dir = data_dir
        self.folder_name = folder_name
        self.img_dir = os.path.join(data_dir, folder_name)
        self.img_labels = pd.read_csv(
            os.path.join(data_dir, f"{folder_name}.csv")
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, str(self.img_labels.iloc[idx, 0]).zfill(5) + ".png"
        )
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        label = [int(i) for i in str(label)]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
