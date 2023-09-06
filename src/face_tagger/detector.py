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

    def detect_face_and_crop(self, image):
        """
        Detects faces in an image and returns cropped face images.

        :param image: Image to detect faces.
        :return: Cropped face images.
        """

        # Face detection using MTCNN.
        faces = self.face_detector.detect_faces(image)

        # Crop face areas in the image.
        cropped_faces = []
        for face in faces:
            x, y, width, height = face['box']
            cropped_face = image[y:y + height, x:x + width]
            cropped_faces.append(cropped_face)

        return cropped_faces
