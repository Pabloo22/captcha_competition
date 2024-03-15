from torch.utils.data import IterableDataset, DataLoader

from captcha_competition.data import generate_captcha_tensors


# pylint: disable=abstract-method, too-few-public-methods
class SyntheticCaptchaIterableDataset(IterableDataset):
    def __iter__(self):
        # This method generates CAPTCHA images and labels indefinitely
        while True:
            captcha_image, label = generate_captcha_tensors()
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
