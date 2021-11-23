import numpy as np
import warnings
from Project.services.dimensionalityReduction import SVDModel


def decompose_svd(matrix, k):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    matrix = np.array(matrix)
    # matrix = np.array([[2,4],[1,3],[0,0],[0,0]])
    # U, S, VT = np.linalg.svd(matrix)
    DDT = matrix @ matrix.T
    DTD = matrix.T @ matrix
    DDT_values, U = np.sqrt(np.real(np.linalg.eig(DDT)[0])), np.real(np.linalg.eig(DDT)[1])
    DTD_values, V = sort_eigen(np.sqrt(np.real(np.linalg.eig(DTD)[0])), np.real(np.linalg.eig(DTD)[1]))
    S = get_core_matrix(matrix, DDT_values, DTD_values)
    return U[:, :k]


def get_core_matrix(matrix, w, wt):
    core = np.zeros((matrix.shape[0], matrix.shape[1]))
    if len(w) <= len(wt):
        S = w
    else:
        S = wt
    for i in range(len(S)):
        core[i][i] = S[i]
    return core


def sort_eigen(eigen_values, eigen_vectors):
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_values, eigen_vectors
