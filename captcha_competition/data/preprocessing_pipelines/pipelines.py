"""Created to avoid git conflicts when working on the same file."""

from typing import Callable
import functools

import numpy as np

from captcha_competition.data.preprocessing_pipelines import (
    remove_background,
    to_grayscale,
    apply_morphological_closing,
    resize,
)


STEPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "remove_background": remove_background,
    "to_grayscale": to_grayscale,
    "closing": apply_morphological_closing,
    "resize_resnet": functools.partial(resize, height=64, width=192),
    "resize_efficientnet": functools.partial(resize, height=80, width=210),
}


def create_preprocessing_pipeline(step_names: list[str]):
    def pipeline(image: np.ndarray) -> np.ndarray:
        for step_name in step_names:
            step = STEPS[step_name]
            image = step(image)
        return image

    return pipeline
