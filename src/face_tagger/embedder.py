import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from .utils import resize_image, convert_bgr_to_rgb


class Embedder:
    """
    Class for face embedding.
    """

    def __init__(self, use_gpu=False):
        """
        Constructor of Embedder class.
        :param use_gpu: Use GPU or not.
        """

        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.ToTensor()

    def prepare_image_tensor(self, face_image):
        """
        Convert the image to tensor format and add batch dimension.
        :param face_image: An image to process.
        :return: Processed image tensor.
        """

        return self.transform(face_image).unsqueeze(0).to(self.device)

    def compute_embeddings(self, face_image):
        """
        Compute embeddings of an image.
        :param face_image: An image to compute embeddings.
        :return: Embeddings of an image.
        """

        face_image = resize_image(face_image, 160, 160)
        face_image = convert_bgr_to_rgb(face_image)
        img_tensor = self.prepare_image_tensor(face_image)

        # Compute embeddings.
        img_embedding = self.resnet(img_tensor)

        return img_embedding.cpu()
