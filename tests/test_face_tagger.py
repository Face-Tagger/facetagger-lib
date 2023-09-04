import unittest

from src.face_tagger.face_tagger import face_tagger


class TestFaceTagger(unittest.TestCase):
    def test_face_tagger(self):
        self.assertEqual(face_tagger(), "Hello Face Tagger")


if __name__ == '__main__':
    unittest.main()
