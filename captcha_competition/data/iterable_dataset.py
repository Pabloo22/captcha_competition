from typing import Callable, Generator

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

from captcha_competition.data import generate_captcha_image
from captcha_competition.data.preprocessing_pipelines import (
    label_to_tensor,
    image_to_tensor,
)


# pylint: disable=abstract-method, too-few-public-methods
class SyntheticCaptchaIterableDataset(IterableDataset):

    def __init__(
        self,
        preprocessing_fc: Callable[[np.ndarray], np.ndarray],
    ):
        super().__init__()
        self.preprocessing_fc = preprocessing_fc

    def __iter__(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        # This method generates CAPTCHA images and labels indefinitely
        while True:
            captcha_image, label = generate_captcha_image()
            captcha_image = self.preprocessing_fc(captcha_image)

            captcha_image_tensor = image_to_tensor(captcha_image)
            label_tensor = label_to_tensor(label)
            yield captcha_image_tensor, label_tensor


if __name__ == "__main__":
    dataset = SyntheticCaptchaIterableDataset(preprocessing_fc=lambda x: x)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
    num_samples = 10

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        print(f"Batch {batch_idx}:")
        print(images.shape, labels.shape)

        if batch_idx >= num_samples:
            break
