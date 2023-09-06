import hdbscan
import torch


class Classifier:
    """
    Class for face classification.
    """

    def __init__(self):
        """
        Constructor of Classifier class.
        """

        pass

    def convert_to_numpy(self, face_embeddings):
        """
        Convert embeddings to numpy format.
        :param face_embeddings: Embeddings to convert.
        :return: Numpy format embeddings.
        """

        return torch.stack(face_embeddings).view(len(face_embeddings), -1).detach().numpy()

    def cluster_embeddings(self, embeddings_np):
        """
        Cluster embeddings using HDBSCAN.
        :param embeddings_np: Numpy format embeddings to cluster.
        :return: Cluster labels.
        """

        hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=2).fit(embeddings_np)

        return hdbscan_cluster.labels_

    def classify_faces(self, face_embeddings):
        """
        Classify faces using clustering.
        :param face_embeddings: Embeddings to classify.
        :return: Cluster labels.
        """

        embeddings_np = self.convert_to_numpy(face_embeddings)

        return self.cluster_embeddings(embeddings_np)
