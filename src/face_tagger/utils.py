import cv2
import numpy as np
from PIL import Image


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
