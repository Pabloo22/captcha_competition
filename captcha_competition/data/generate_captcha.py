import os
import random
import functools
from pathlib import Path
import numpy as np

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from captcha_competition import FONTS_PATH
from captcha_competition.data.preprocessing import (
    label_to_tensor,
    image_to_tensor,
)


WIDTH = 200
HEIGHT = 80
NUM_NUMBERS = 6
Y_CENTER = HEIGHT // 2
FONT_SIZE = 64
TENSOR_TYPE = torch.float32

DIFF_BETWEEN_BACKGROUND_AND_NUMBER_COLORS = 40

NUMBER_OF_RANDOM_DOTS = (50, 100)
NUMBER_OF_RANDOM_LINES = (10, 20)
NUMBER_OF_RANDOM_CIRCLES = (1, 5)


def generate_captcha_image() -> tuple[np.ndarray, list[int]]:
    # Create a blank image with white background
    background_color = tuple(np.random.choice(range(256), size=3))
    image = Image.new("RGB", (WIDTH, HEIGHT), color=background_color)  # type: ignore
    draw = ImageDraw.Draw(image)

    segment_width = WIDTH // NUM_NUMBERS
    numbers = []

    font = load_random_font()
    numbers_color = tuple(np.random.choice(range(256), size=3))
    while (
        np.abs(np.array(numbers_color) - np.array(background_color)).sum()
        < DIFF_BETWEEN_BACKGROUND_AND_NUMBER_COLORS
    ):
        numbers_color = tuple(np.random.choice(range(256), size=3))

    y = 0
    # Generate and draw random numbers
    for i in range(NUM_NUMBERS):
        number = random.randint(0, 9)
        numbers.append(number)
        x = i * segment_width
        draw.text((x, y), str(number), fill=numbers_color, font=font)  # type: ignore

    # Add random dots
    for _ in range(random.randint(*NUMBER_OF_RANDOM_DOTS)):
        position = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        numbers_color = tuple(np.random.choice(range(256), size=3))
        draw.point(position, fill=numbers_color)  # type: ignore

    # Add random lines
    for _ in range(random.randint(*NUMBER_OF_RANDOM_LINES)):
        start_position = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        end_position = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        numbers_color = tuple(np.random.choice(range(256), size=3))
        draw.line([start_position, end_position], fill=numbers_color, width=1)  # type: ignore

    # Add random circles
    for _ in range(random.randint(*NUMBER_OF_RANDOM_CIRCLES)):
        top_left = (
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT - 20),
        )
        bottom_right = (
            top_left[0] + random.randint(10, 30),
            top_left[1] + random.randint(10, 30),
        )
        numbers_color = tuple(np.random.choice(range(256), size=3))
        draw.ellipse(
            [top_left, bottom_right], outline=numbers_color, fill=numbers_color  # type: ignore
        )

    return np.array(image), numbers


def load_random_font(
    fonts_dir=FONTS_PATH, size=FONT_SIZE
) -> ImageFont.FreeTypeFont:
    font_files = list_all_fonts(fonts_dir)
    selected_font_file = random.choice(font_files)
    font = ImageFont.truetype(str(selected_font_file), size=size)
    return font


@functools.cache
def list_all_fonts(fonts_dir=FONTS_PATH) -> list[Path]:
    # We should list all the font files in the specified directory or
    # in directories under the specified directory (recursively)
    font_files = []
    for root, _, files in os.walk(fonts_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".otf"):
                font_files.append(Path(root) / file)
    return font_files


if __name__ == "__main__":
    captcha_image, numbers_ = generate_captcha_image()
    captcha_image.show()
    print(numbers_)
