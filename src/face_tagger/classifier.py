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

    def classify_faces(self, face_embeddings):
        """
        Cluster embeddings using HDBSCAN.

        :param face_embeddings: Embeddings to cluster.
        :return: Cluster labels.
        """

        # Cluster embeddings using HDBSCAN.
        embeddings_np = torch.stack(face_embeddings).view(len(face_embeddings), -1).detach().numpy()
        hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=2).fit(embeddings_np)

        return hdbscan_cluster.labels_
