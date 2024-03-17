import torch

import numpy as np
import cv2


TENSOR_TYPE = torch.float32


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image_channel_first = np.transpose(image, (2, 0, 1))
    return torch.tensor(image_channel_first.tolist(), dtype=TENSOR_TYPE)


def label_to_tensor(label: list[int]) -> torch.Tensor:
    return torch.tensor(label, dtype=TENSOR_TYPE)


def resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(img, (width, height))


def to_grayscale(img: np.ndarray) -> np.ndarray:
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.convertScaleAbs(new_img)
    return new_img


def apply_morphological_closing(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def remove_background(img: np.ndarray) -> np.ndarray:
    numbers_color = get_numbers_color(img)
    mask = np.all(img == numbers_color, axis=-1)
    mask = np.stack([mask, mask, mask], axis=-1)
    new_img = img * mask

    return new_img


# --- Helper functions ---


def get_numbers_color(img: np.ndarray):
    background_color = get_background_color(img)
    colors, _ = get_most_common_colors(img, n=2)

    first_color, second_color = colors
    if np.all(first_color == background_color):
        return second_color
    return first_color


def get_background_color(img: np.ndarray):
    height = img.shape[0]
    top = img[: height // 4]
    bottom = img[3 * height // 4 :]

    top_color, top_count = get_most_common_colors(top, n=1)
    bottom_color, bottom_count = get_most_common_colors(bottom, n=1)

    if top_count[0] > bottom_count[0]:
        return top_color[0]
    return bottom_color[0]


def get_most_common_colors(img: np.ndarray, n: int = 1):
    """Finds the n most common colors in the image.

    Parameters:
    - img: The image array.
    - n: The number of top common colors to return.

    Returns:
    - A tuple of two arrays:
        - The first array contains the top n most common colors.
        - The second array contains the counts of these colors.
    """
    unique, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    # Ensure n does not exceed the number of unique colors
    n = min(n, len(unique))
    sorted_indices = np.argsort(counts)[-n:]
    return unique[sorted_indices][::-1], counts[sorted_indices][::-1]
