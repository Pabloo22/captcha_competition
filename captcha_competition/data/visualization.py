from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_image(image_array: np.ndarray, use_cvt_color: bool = True) -> Figure:
    """Plots an image from a numpy array."""
    plt.figure(figsize=(5, 5))
    new_image_array = image_array
    if use_cvt_color:
        new_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    plt.imshow(new_image_array)
    plt.axis("off")

    return plt.gcf()
