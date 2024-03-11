from torch.utils.data import IterableDataset, DataLoader
import torch
from PIL import Image
import numpy as np

from captcha_competition import generate_captcha_image


# pylint: disable=abstract-method
class SyntheticCaptchaIterableDataset(IterableDataset):
    def __iter__(self):
        # This method generates CAPTCHA images and labels indefinitely
        while True:
            captcha_image, label = self.generate_captcha()
            yield captcha_image, label

    def generate_captcha(self):
        while True:
            image, label = generate_captcha_image()

            # Convert the label to a tensor
            label = self.label_to_tensor(label)
            image = self.image_to_tensor(image)
            return image, label

    def label_to_tensor(self, label: list[int]):
        # TODO: Implement this method
        return torch.tensor(label)

    def image_to_tensor(self, image: Image.Image):
        return torch.tensor(np.array(image).tolist())


if __name__ == "__main__":
    dataset = SyntheticCaptchaIterableDataset()
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
    num_samples = 10

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        print(f"Batch {batch_idx}:")
        print(images.shape, labels.shape)

        if batch_idx >= num_samples:
            break
