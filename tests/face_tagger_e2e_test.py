import os
import unittest

import cv2

from src.face_tagger import FaceTagger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestFaceTagger(unittest.TestCase):
    def setUp(self):
        self.face_tagger = FaceTagger(use_gpu=False, min_faces_to_identify_human=2, min_similarity_face_count=1)

    def test_classify_images_by_person(self):
        image_objects = []

        image_data_map = {
            "one_person_1.jpg": 3,
            "one_person_3.jpg": 2,
            "two_people.jpg": 1
        }

        for image_name, count in image_data_map.items():
            image_data = cv2.imread(os.path.join(BASE_DIR, "resources", image_name))

            for i in range(count):
                image_obj = type("ImageObject", (object,), {
                    "image_data": image_data,
                    "image_id": f"{image_name.split('.')[0]}_{i + 1}"
                })
                image_objects.append(image_obj)

        result = self.face_tagger.classify_images_by_person(image_objects)

        self.assertIsInstance(result, dict)
        self.assertIn("unclassified_images", result)
        self.assertEqual(len(result["unclassified_images"]), 1)
        self.assertEqual(len(result["group_1"]["others"]), 2)
        self.assertEqual(len(result["group_2"]["others"]), 1)
