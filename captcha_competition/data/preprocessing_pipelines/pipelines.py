"""Created to avoid git conflicts when working on the same file."""

from typing import Callable, Any
import functools

from captcha_competition.data.preprocessing_pipelines import (
    remove_background,
    to_grayscale,
    min_max_normalize,
    apply_morphological_closing,
    resize,
    image_to_tensor,
    remove_background_tensor,
    to_grayscale_tensor,
    min_max_normalize_tensor,
    resize_tensor,
)


STEPS: dict[str, Callable[[Any], Any]] = {
    "remove_background": remove_background,
    "to_grayscale": to_grayscale,
    "closing": apply_morphological_closing,
    "min_max_normalize": min_max_normalize,
    "resize_resnet": functools.partial(resize, height=64, width=192),
    "resize_efficientnet": functools.partial(resize, height=80, width=210),
    "image_to_tensor": image_to_tensor,
    "remove_background_tensor": remove_background_tensor,
    "to_grayscale_tensor": to_grayscale_tensor,
    "min_max_normalize_tensor": min_max_normalize_tensor,
    "resize_resnet_tensor": functools.partial(
        resize_tensor, height=64, width=192
    ),
    "resize_efficientnet_tensor": functools.partial(
        resize_tensor, height=80, width=210
    ),
}


def create_preprocessing_pipeline(step_names: list[str]):
    def pipeline(image: Any) -> Any:
        for step_name in step_names:
            step = STEPS[step_name]
            image = step(image)
        return image

    return pipeline
