import numpy as np
import cv2


def most_common_colors(img, n=1):
    unique, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    return unique[np.argsort(counts)[-n]], counts[np.argsort(counts)[-n]]


def get_bg_color(img):
    top = img[: int(img.shape[0] / 4)]
    bottom = img[int(img.shape[0] * 3 / 4) :]
    top_color, top_count = most_common_colors(top)
    bottom_color, bottom_count = most_common_colors(bottom)
    if top_count > bottom_count:
        return top_color
    else:
        return bottom_color


def remove_bg_v1(img, grayscale=True):
    bg_color = get_bg_color(img)
    first_color, _ = most_common_colors(img, n=1)
    second_color, _ = most_common_colors(img, n=2)
    if np.all(first_color == bg_color):
        num_color = second_color
    else:
        num_color = first_color
    mask = np.all(img == num_color, axis=-1)
    mask = np.stack([mask, mask, mask], axis=-1)
    new_img = img * mask
    if grayscale:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        new_img = cv2.convertScaleAbs(new_img)
    return new_img