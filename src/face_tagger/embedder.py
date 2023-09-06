import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1


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

    def compute_embeddings(self, face_image):
        """
        Compute embeddings of an image.

        :param face_image: An image to compute embeddings.
        :return: Embeddings of an image.
        """

        # Resize the image to 160x160.
        face_image = cv2.resize(face_image, (160, 160))

        # Convert the image to RGB.
        face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        # Convert the image to tensor and add a batch dimension.
        img_tensor = self.transform(face_image).unsqueeze(0).to(self.device)

        # Compute embeddings.
        img_embedding = self.resnet(img_tensor)

        return img_embedding.cpu()
