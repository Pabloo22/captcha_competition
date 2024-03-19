import os
import random
import functools
from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import torch

from captcha_competition import FONTS_PATH


WIDTH = 200
HEIGHT = 80
NUM_NUMBERS = 6
FONT_SIZE = 64
TENSOR_TYPE = torch.float32

Y_MAXIMUM_OFFSET = 10

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

    y = random.randint(-Y_MAXIMUM_OFFSET, Y_MAXIMUM_OFFSET)
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
    from captcha_competition import DATA_RAW_PATH
    import pandas as pd
    import tqdm  # type: ignore

    num_images = 1_000_000
    labels: list[tuple[int, str]] = []
    # Create DATA_RAW_PATH / "synthetic" directory if it does not exis
    # DATA_RAW_PATH = Path(
    #     "/home/pablo/VSCodeProjects/captcha_competition/data/raw/"
    # )
    print(DATA_RAW_PATH / "synthetic")
    os.makedirs(DATA_RAW_PATH / "synthetic", exist_ok=True)

    try:
        for i in tqdm.trange(num_images):
            image_, numbers_ = generate_captcha_image()
            image_path = DATA_RAW_PATH / "synthetic" / f"{i:07}.png"
            Image.fromarray(image_).save(image_path)
            labels.append((i, "".join(map(str, numbers_))))
    finally:
        labels_df = pd.DataFrame(labels, columns=["Id", "Label"])
        labels_df.to_csv(DATA_RAW_PATH / "synthetic.csv", index=False)
