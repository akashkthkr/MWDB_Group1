# getter for getting clusters
import random
import numpy as np

max_iter = 300


def get_clusters_top_k(feature_matrix, k):
    # centroids = np.random.rand(k, feature_matrix.shape[1])
    shapeSize = np.array(feature_matrix).shape
    centroids = np.random.rand(k, shapeSize[1])

    labels = np.zeros((shapeSize[0], 1), dtype=np.int64)
    clusterSize = np.zeros((k, 1), dtype=np.int64)
    for i in range(0, k):
        rand_int = random.randint(0, shapeSize[0] - 1)
        print("value of i ", i)
        print("value of random int ", rand_int)
        print("feature matrix", feature_matrix[rand_int])
        centroids[i] = feature_matrix[rand_int]
    for i in range(0, max_iter):
        clusterSize = np.zeros((k, 1), dtype=np.int64)
        for j in range(0, shapeSize[0]):

            euclidDist = np.empty((k))
            for l in range(0, k):
                euclidDist[l] = np.linalg.norm(feature_matrix[j] - centroids[l])

            labels[j] = int(np.argmin(euclidDist))
            clusterSize[int(labels[j])] += 1

        for l in range(0, k):
            if (clusterSize[l] != 0):
                centroids[l] = 0.0

        for j in range(0, shapeSize[0]):
            centroids[int(labels[j])] += (feature_matrix[j] / clusterSize[int(labels[j])])

    resultant_feature_mat = np.empty(((shapeSize[0], k)))
    for i in range(0, shapeSize[0]):
        euclidDist = []
        for j in range(0, k):
            euclidDist.append(np.linalg.norm(feature_matrix[i] - centroids[j]))
        resultant_feature_mat[i] = np.array(euclidDist)
    return resultant_feature_mat


'''
let hte imge x
x - centrond
avg ( feature -- cc)
(rest x_result)
dist()'''
