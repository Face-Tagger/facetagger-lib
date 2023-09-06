import cv2

from src.face_tagger.classifier import Classifier
from src.face_tagger.detector import Detector
from src.face_tagger.embedder import Embedder


class FaceTagger:
    """
    Main class of the face tagger module.
    """

    def __init__(self, use_gpu=False, image_resize_factor=1.0):
        """
        Constructor of FaceTagger class.

        :param use_gpu: Use GPU or not.
        :param image_resize_factor: Resize factor of images.
        """

        self.detector = Detector()
        self.embedder = Embedder(use_gpu)
        self.classifier = Classifier()
        self.image_resize_factor = image_resize_factor

    def classify_images_by_person(self, image_objects):
        """
        Classifies images based on the person in the image.

        :param image_objects: Images to classify.
        :return: Classified images.
        """

        face_embeddings = []
        processed_image_ids = []
        face_counts_per_image = {}

        classified_images = {"unclassified_images": []}

        # Embedding faces in each image.
        for image_object in image_objects:
            if self.image_resize_factor != 1.0:
                image = cv2.resize(image_object.image_data, (
                    int(image_object.image_data.shape[1] * self.image_resize_factor),
                    int(image_object.image_data.shape[0] * self.image_resize_factor)))
            else:
                image = image_object.image_data

            if image is not None:
                cropped_faces = self.detector.detect_face_and_crop(image)

                if len(cropped_faces) == 0:
                    # If there is no face in the image, add the image to the list of unclassified images.
                    classified_images["unclassified_images"].append(image_object.image_id)
                    continue

                face_counts_per_image[image_object.image_id] = len(cropped_faces)

                # Compute embeddings of cropped faces.
                for cropped_face in cropped_faces:
                    face_embedding = self.embedder.compute_embeddings(cropped_face)
                    face_embeddings.append(face_embedding)
                    processed_image_ids.append(image_object.image_id)

        # Classifying embedded faces.
        groups = self.classifier.classify_faces(face_embeddings)

        # Grouping images by person.
        for group, image_id in zip(groups, processed_image_ids):
            if group != -1:
                group_key = f"group_{group + 1}"

                if group_key not in classified_images:
                    # If the group is not in the dictionary, add the group.
                    classified_images[group_key] = {"main": None, "others": []}

                if face_counts_per_image[image_id] == 1:
                    # If there is only one face in the image, the face is the main face.
                    classified_images[group_key]["main"] = image_id
                    continue

                if image_id not in classified_images[group_key]["others"]:
                    # If the image is not in the list of other images, add the image.
                    classified_images[group_key]["others"].append(image_id)

        return classified_images
