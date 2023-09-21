import os
import sys

from .classifier import Classifier
from .detector import Detector
from .embedder import Embedder
from .utils import resize_image


class FaceTagger:
    """
    Main class of the face tagger module.
    """

    def __init__(self, use_gpu=False, image_resize_factor=1.0, min_faces_to_identify_human=3, min_similarity_face_count=2):
        """
        Constructor of FaceTagger class.
        :param use_gpu: Use GPU or not.
        :param image_resize_factor: Resize factor of images.
        :param min_faces_to_identify_human: Minimum number of faces required to be classified as a human.
        :param min_similarity_face_count: Minimum number of similar faces required to be included in a group.
        """

        self.detector = Detector()
        self.embedder = Embedder(use_gpu)
        self.classifier = Classifier(min_faces_to_identify_human=min_faces_to_identify_human,
                                     min_similarity_face_count=min_similarity_face_count)
        self.image_resize_factor = image_resize_factor

    def detect_and_embed_faces(self, image):
        """
        Detects and embeds faces in an image.
        :param image: Image to detect and embed faces.
        :return: Cropped faces and their embeddings.
        """

        face_embeddings = []
        cropped_faces = self.detector.detect_face_and_crop(image)

        for cropped_face in cropped_faces:
            face_embedding = self.embedder.compute_embeddings(cropped_face)
            face_embeddings.append(face_embedding)

        return cropped_faces, face_embeddings

    def group_and_classify(self, face_embeddings, processed_image_ids):
        """
        Groups and classifies images based on the person in the image.
        :param face_embeddings: Embeddings of faces.
        :param processed_image_ids: Image ids of the processed images.
        :return: Classified images.
        """

        classified_images = {"unclassified_images": []}
        groups = self.classifier.classify_faces(face_embeddings)

        face_counts_per_image = {}
        for image_id in processed_image_ids:
            face_counts_per_image[image_id] = face_counts_per_image.get(image_id, 0) + 1

        unclassified_image_ids_set = set(processed_image_ids)

        for group, image_id in zip(groups, processed_image_ids):
            if group != -1:
                group_key = f"group_{group + 1}"

                if group_key not in classified_images:
                    classified_images[group_key] = {"main": None, "others": []}

                if face_counts_per_image[image_id] == 1 and classified_images[group_key]["main"] is None:
                    classified_images[group_key]["main"] = image_id
                else:
                    if image_id not in classified_images[group_key]["others"]:
                        classified_images[group_key]["others"].append(image_id)

                unclassified_image_ids_set.discard(image_id)

        classified_images["unclassified_images"].extend(list(unclassified_image_ids_set))

        return classified_images

    def classify_images_by_person(self, image_objects):
        """
        Classifies images based on the person in the image.
        :param image_objects: Images to classify.
        :return: Classified images.
        """

        face_embeddings = []
        processed_image_ids = []
        unclassified_images = []

        sys.stdout = open(os.devnull, 'w')

        # Embedding faces in each image.
        for image_object in image_objects:
            if self.image_resize_factor != 1.0:
                image = resize_image(image_object.image_data,
                                     int(image_object.image_data.shape[1] * self.image_resize_factor),
                                     int(image_object.image_data.shape[0] * self.image_resize_factor))
            else:
                image = image_object.image_data

            if image is not None:
                cropped_faces, current_face_embeddings = self.detect_and_embed_faces(image)
                face_embeddings.extend(current_face_embeddings)

                if len(cropped_faces) == 0:
                    # If there is no face in the image, add the image to the list of unclassified images.
                    unclassified_images.append(image_object.image_id)
                    continue

                processed_image_ids.extend([image_object.image_id] * len(cropped_faces))

        sys.stdout = sys.__stdout__

        classified_images = self.group_and_classify(face_embeddings, processed_image_ids)
        classified_images["unclassified_images"] = list(
            set(classified_images["unclassified_images"]).union(set(unclassified_images)))

        return classified_images
