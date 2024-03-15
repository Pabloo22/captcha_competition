from torch.utils.data import IterableDataset, DataLoader
import numpy as np

from captcha_competition.data import (
    generate_captcha_image,
    label_to_tensor,
    image_to_tensor,
)


# pylint: disable=abstract-method, too-few-public-methods
class SyntheticCaptchaIterableDataset(IterableDataset):
    def __init__(self, preprocessing_fc=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessing_fc = preprocessing_fc

    def __iter__(self):
        # This method generates CAPTCHA images and labels indefinitely
        while True:
            captcha_image, label = generate_captcha_image()
            if self.preprocessing_fc is not None:
                captcha_image = self.preprocessing_fc(captcha_image)
            captcha_image = image_to_tensor(captcha_image)
            label = label_to_tensor(label)
            yield captcha_image, label


if __name__ == "__main__":
    dataset = SyntheticCaptchaIterableDataset()
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
    num_samples = 10

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        print(f"Batch {batch_idx}:")
        print(images.shape, labels.shape)

        if batch_idx >= num_samples:
            break
