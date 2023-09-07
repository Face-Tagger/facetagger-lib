import os

import cv2
import numpy as np
from PIL import Image

from .models import ImageObject


def resize_image(image, width, height):
    """
    Resize the given image to specified width and height.
    :param image: An image to resize.
    :param width: Width of the resized image.
    :param height: Height of the resized image.
    :return: Resized image.
    """

    return cv2.resize(image, (width, height))


def convert_bgr_to_rgb(image):
    """
    Convert the BGR format image to RGB format using OpenCV.
    :param image: An image to convert.
    :return: RGB image.
    """

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def bytes_to_cvimage(byte_stream):
    """
    Convert byte stream to OpenCV image format.
    :param byte_stream: Bytes representing image data.
    :return: OpenCV format image.
    """

    return cv2.imdecode(np.frombuffer(byte_stream, np.uint8), cv2.IMREAD_COLOR)


def load_images_from_directory(images_path):
    """
    Generate image objects from the given directory.
    :param images_path: Directory containing images.
    :return: ImageObject generator.
    """
    for filename in os.listdir(images_path):
        image_data = cv2.imread(os.path.join(images_path, filename))
        if image_data is not None:
            yield ImageObject(filename, image_data)


def load_image(image_path):
    """
    Load image from the given path.
    :param image_path: Path to the image.
    :return: ImageObject.
    """
    image_data = cv2.imread(image_path)
    return ImageObject(image_path, image_data)
