from mtcnn import MTCNN


class Detector:
    """
    Class for face detection.
    """

    def __init__(self):
        """
        Constructor of Detector class.
        """

        self.face_detector = MTCNN()

    def detect_face(self, image):
        """
        Detects faces in an image.
        :param image: Image to detect faces.
        :return: Detected faces with their bounding boxes.
        """

        # Face detection using MTCNN.
        return self.face_detector.detect_faces(image)

    def crop(self, image, face):
        """
        Crops the face area from the image.
        :param image: Image from which to crop the face.
        :param face: Detected face with bounding box.
        :return: Cropped face image.
        """

        x, y, width, height = face['box']

        return image[y:y + height, x:x + width]

    def detect_face_and_crop(self, image):
        """
        Detects faces in an image and returns cropped face images.
        :param image: Image to detect faces.
        :return: List of cropped face images.
        """

        faces = self.detect_face(image)

        return [self.crop(image, face) for face in faces]
