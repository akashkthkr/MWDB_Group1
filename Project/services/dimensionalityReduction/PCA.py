import numpy as np


def get_top_k_eigen_decomposition(matrix, k):
    eigen_val, eigen_vect = np.linalg.eig(matrix)
    eigen_val = eigen_val.real
    eigen_vect = eigen_vect.real

    idx = np.argsort(eigen_val)[::1]
    eigen_vect = eigen_vect[:, idx]
    eigen_val = eigen_val[idx]
    eigen_vect = eigen_vect[:, 0:k]
    eigen_val = eigen_vect[:k]
    return (eigen_vect, eigen_val)

def get_PCA(features, top_K_principal_components):
    print(features.shape)
    mean_vec = np.mean(features, axis=0)
    covariance_matrix =  (features - mean_vec).T.dot((features - mean_vec)) / (features.shape[0 ] -1)

    # get eigen val and vect for
    eigen_vects, eigen_vals = get_top_k_eigen_decomposition(covariance_matrix, top_K_principal_components)
    # return np.dot(eigen_vects.T, features.T).T, (np.diag(eigen_vals)), eigen_vects.T
    return np.dot(eigen_vects.T, features.T).T