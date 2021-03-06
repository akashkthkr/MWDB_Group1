import numpy as np
from Project.services.dimensionalityReduction import SVDModel, PCA, kmeans


def get_features(features, model_name, k):
    if model_name == "PCA":
        matrix = PCA.get_PCA(np.array(features), k)
    elif model_name == "SVD":
        matrix = SVDModel.get_svd(features, k)
    elif model_name == "KMeans":
        matrix = kmeans.get_clusters_top_k(features, k)
    else:
        return None

    return matrix
