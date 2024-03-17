import numpy as np
import cv2
import torch
from captcha_competition.data.generate_captcha import generate_captcha_image
import matplotlib.pyplot as plt

TENSOR_TYPE = torch.float32
NUM_NUMBERS = 6


############ Preprocessing Type 1 ############


def generate_captcha_tensors(
    preprocessing_fc=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, label = generate_captcha_image()
    tensor_label = label_to_tensor(label)
    if preprocessing_fc is not None:
        image = preprocessing_fc(image)
    tensor_image = image_to_tensor(image)
    return tensor_image, tensor_label


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


def label_to_tensor(label: list[int]) -> torch.Tensor:
    return torch.tensor(label, dtype=TENSOR_TYPE)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image_channel_first = np.transpose(image, (2, 0, 1))
    return torch.tensor(image_channel_first.tolist(), dtype=TENSOR_TYPE)


def remove_bg_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = remove_bg_v1(image)
    return image_to_tensor(image)


############ Preprocessing Type 2 ############


def cleaned_image(image: np.ndarray, kernel_size=2) -> np.ndarray:
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize
    # Top most left pixel is the background color
    background_color = image[0,0]
    # If the background color is dark and the text is light
    if background_color < 128:
        ret, binary = cv2.threshold(image, background_color, 255, cv2.THRESH_BINARY)
    # If the background color is light and the text is dark
    else:
        ret, binary = cv2.threshold(image, background_color-1, 255, cv2.THRESH_BINARY_INV)
    
     # Close the image: dilate and then erode to remove dots and lines
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def cleaned_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = cleaned_image(image)
    return image_to_tensor(image)

############ Preprocessing Type 3 ############

def preprocessing_image(image: np.ndarray, kernel_size=5) -> np.ndarray:
    
    # Best of preprocessing type 1
    image = remove_bg_v1(image)
    # Best of preprocessing type 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return eroded

def preprocessing_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = preprocessing_image(image)
    return image_to_tensor(image)

############ Testing ############

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_, label_ = generate_captcha_image()
    new_image = remove_bg_v1(image_)
    new_image2 = cleaned_image(image_)
    new_image3 = preprocessing_image(image_)
    
    plt.figure(figsize=(10, 6))

    plt.subplot(141), plt.imshow(image_, cmap='gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(142), plt.imshow(new_image, cmap='gray')
    plt.title('Preprocessed Mode 1')
    plt.xticks([]), plt.yticks([])

    plt.subplot(143), plt.imshow(new_image2, cmap='gray')
    plt.title('Preprocessed Mode 2')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(144), plt.imshow(new_image3, cmap='gray')
    plt.title('Best of both worlds')
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()