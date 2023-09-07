import os
import sys
import unittest

import cv2
import numpy as np
import torch

from src.face_tagger import FaceTagger
from src.face_tagger.classifier import Classifier
from src.face_tagger.detector import Detector
from src.face_tagger.embedder import Embedder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = Embedder(use_gpu=False)
        self.sample_image = cv2.imread(os.path.join(BASE_DIR, "resources", "one_person_1.jpg"))

    def test_prepare_image_tensor(self):
        tensor = self.embedder.prepare_image_tensor(self.sample_image)
        self.assertIsInstance(tensor, torch.Tensor)

    def test_compute_embeddings(self):
        embedding = self.embedder.compute_embeddings(self.sample_image)
        self.assertIsInstance(embedding, torch.Tensor)


class TestDetector(unittest.TestCase):

    def setUp(self):
        self.detector = Detector()

    def test_detect_one_face(self):
        test_cases_one = [
            os.path.join(BASE_DIR, "resources", "one_person_1.jpg"),
            os.path.join(BASE_DIR, "resources", "one_person_2.jpg"),
            os.path.join(BASE_DIR, "resources", "one_person_3.jpg"),
            os.path.join(BASE_DIR, "resources", "one_person_4.jpg"),
            os.path.join(BASE_DIR, "resources", "one_person_5.jpg"),
            os.path.join(BASE_DIR, "resources", "one_person_6.jpg")
        ]

        test_cases_multiple = {
            os.path.join(BASE_DIR, "resources", "two_people.jpg"): 2,
            os.path.join(BASE_DIR, "resources", "seven_people.jpg"): 7
        }

        for file_path in test_cases_one:
            sys.stdout = open(os.devnull, 'w')
            face = self.detector.detect_face(cv2.imread(file_path))
            sys.stdout = sys.__stdout__
            self.assertEqual(len(face), 1)

        for file_path, expected_faces in test_cases_multiple.items():
            sys.stdout = open(os.devnull, 'w')
            faces = self.detector.detect_face(cv2.imread(file_path))
            sys.stdout = sys.__stdout__
            self.assertEqual(len(faces), expected_faces)


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = Classifier()
        self.sample_embeddings = [torch.rand((512,)) for _ in range(5)]

    def test_convert_to_numpy(self):
        np_embeddings = self.classifier.convert_to_numpy(self.sample_embeddings)
        self.assertIsInstance(np_embeddings, np.ndarray)

    def test_cluster_embeddings(self):
        np_embeddings = self.classifier.convert_to_numpy(self.sample_embeddings)
        labels = self.classifier.cluster_embeddings(np_embeddings)
        self.assertEqual(len(labels), len(self.sample_embeddings))

    def test_classify_faces(self):
        labels = self.classifier.classify_faces(self.sample_embeddings)
        self.assertEqual(len(labels), len(self.sample_embeddings))


class TestFaceTagger(unittest.TestCase):

    def setUp(self):
        self.face_tagger = FaceTagger(use_gpu=False)

    def test_detect_and_embed_single_faces(self):
        for i in range(1, 3):
            image = cv2.imread(os.path.join(BASE_DIR, "resources", f"one_person_{i}.jpg"))
            cropped_faces, face_embeddings = self.face_tagger.detect_and_embed_faces(image)
            self.assertEqual(len(cropped_faces), 1)
            self.assertEqual(len(face_embeddings), 1)

    def test_detect_and_embed_multiple_faces(self):
        test_cases = {
            os.path.join(BASE_DIR, "resources", "two_people.jpg"): 2,
            os.path.join(BASE_DIR, "resources", "seven_people.jpg"): 7,
        }

        for file_path, expected_faces in test_cases.items():
            image = cv2.imread(file_path)
            cropped_faces, face_embeddings = self.face_tagger.detect_and_embed_faces(image)
            self.assertEqual(len(cropped_faces), expected_faces)
            self.assertEqual(len(face_embeddings), expected_faces)
