import torch
import torch.functional as F

import numpy as np
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TENSOR_TYPE = torch.float32


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if not is_channel_first(image):
        image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image.tolist(), dtype=TENSOR_TYPE, device=DEVICE)


def is_channel_first(image: np.ndarray) -> bool:
    assert len(image.shape) == 3
    return image.shape[0] < image.shape[2]


def label_to_tensor(label: list[int]) -> torch.Tensor:
    return torch.tensor(label, dtype=torch.long, device=DEVICE)


def resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    # print(f"Resizing image from {img.shape} to ({height}, {width})")
    new_image = cv2.resize(img, (width, height))
    # print(f"Resized image shape: {new_image.shape}")
    return new_image


def resize_tensor(
    tensor: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    # print(f"Resizing tensor from {tensor.shape} to ({height}, {width})")
    new_tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    new_tensor = new_tensor.squeeze(0)
    return new_tensor


def to_grayscale(img: np.ndarray) -> np.ndarray:
    # print(f"Converting image to grayscale with shape {img.shape}")
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.convertScaleAbs(new_img)
    new_img = new_img.reshape(*new_img.shape, 1)
    # Stack the grayscale image 3 times to match the original shape
    new_img = np.repeat(new_img, 3, axis=-1)
    return new_img


def to_grayscale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # print(f"Converting tensor to grayscale with shape {tensor.shape}")
    grayscale_image = F.rgb_to_grayscale(tensor)
    # Stack the grayscale image 3 times to match the original shape
    # Use pytorch's repeat function to avoid copying the data
    new_tensor = grayscale_image.repeat(3, 1)
    return new_tensor


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values using min-max scaling to [0, 1] range."""
    # Ensure image is a float type before normalization
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    # Avoid division by zero in case the image has a constant value
    if max_val - min_val != 0:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = image - min_val
    return normalized_image


def min_max_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor pixel values using min-max scaling to [0, 1] range."""
    # Ensure tensor is a float type before normalization
    tensor = tensor.float()
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # Avoid division by zero in case the tensor has a constant value
    if max_val - min_val != 0:
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
    else:
        normalized_tensor = tensor - min_val
    return normalized_tensor


def apply_morphological_closing(img: np.ndarray) -> np.ndarray:
    # print(f"Applying morphological closing to image with shape {img.shape}")
    kernel = np.ones((3, 3), np.uint8)
    new_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # print("Output shape:", new_img.shape)
    return new_img


def apply_morphological_closing_tensor(tensor: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(3, 3)
    new_tensor = F.morphology(tensor, kernel, "closing")
    return new_tensor


def remove_background(img: np.ndarray) -> np.ndarray:
    numbers_color = get_numbers_color(img)
    mask = np.all(img == numbers_color, axis=-1)
    mask = np.stack([mask, mask, mask], axis=-1)
    new_img = img * mask

    return new_img


def remove_background_tensor(tensor: torch.Tensor) -> torch.Tensor:
    numbers_color = get_numbers_color_tensor(tensor)
    mask = torch.all(tensor == numbers_color, dim=0)
    mask = torch.stack([mask, mask, mask], dim=0)
    new_tensor = tensor * mask

    return new_tensor


# --- Helper functions ---


def get_numbers_color(img: np.ndarray):
    background_color = get_background_color(img)
    colors, _ = get_most_common_colors(img, n=2)

    first_color, second_color = colors
    if np.all(first_color == background_color):
        return second_color
    return first_color


def get_numbers_color_tensor(tensor: torch.Tensor):
    background_color = get_background_color_tensor(tensor)
    colors, _ = get_most_common_colors_tensor(tensor, n=2)

    first_color, second_color = colors
    if torch.all(first_color == background_color):
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


def get_background_color_tensor(tensor: torch.Tensor):
    height = tensor.shape[1]
    top = tensor[:, : height // 4]
    bottom = tensor[:, 3 * height // 4 :]

    top_color, top_count = get_most_common_colors_tensor(top, n=1)
    bottom_color, bottom_count = get_most_common_colors_tensor(bottom, n=1)

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


def get_most_common_colors_tensor(tensor: torch.Tensor, n: int = 1):
    """Finds the n most common colors in the tensor.

    Parameters:
    - tensor: The tensor array.
    - n: The number of top common colors to return.

    Returns:
    - A tuple of two arrays:
        - The first array contains the top n most common colors.
        - The second array contains the counts of these colors.
    """
    unique, counts = torch.unique(
        tensor.reshape(3, -1), dim=1, return_counts=True
    )
    # Ensure n does not exceed the number of unique colors
    n = min(n, len(unique))
    sorted_indices = torch.argsort(counts)[-n:]
    return unique[:, sorted_indices].T.flip(0), counts[sorted_indices].flip(0)
