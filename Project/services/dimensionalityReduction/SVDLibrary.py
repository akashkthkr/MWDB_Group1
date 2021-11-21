import numpy as np
from Project.utilities import MergeFeatures
from sklearn.decomposition import TruncatedSVD
from Project.services.dimensionalityReduction import SVD


def get_svd(features, k):
    features = np.array(features)
    # color_moment = MergeFeatures.get_merged_feature_list(features)
    # feature = np.array(color_moment['image-0']).reshape(3,64)
    # svd = TruncatedSVD(n_components=2)
    # X_reduced = svd.fit_transform(color_moment['image-1'])
    # U, S, VT = scratch_svd.decompose_svd(feature)
    U, S, VT = construct_svd(features, k)
    return U, S, VT

def construct_svd(matrix, n_component):
    U, D, VT = np.linalg.svd(matrix)
    # S = np.zeros((matrix.shape[0], matrix.shape[1]))
    S = np.zeros((len(D), len(D)))
    # S[:matrix.shape[0], :matrix.shape[0]] = np.diag(D)
    S[:len(D), :len(D)] = np.diag(D)
    S = S[:n_component, :n_component]
    VT = VT[:n_component, :]
    U = U[:, :n_component]
    return U, S, VT
    # U.dot(S.dot(VT))