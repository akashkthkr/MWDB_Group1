import pickle
import numpy as np
from Project.Lda.code import dataset


def kl_divergence(p, q):
    epsilon = 0.000001
    p=p+epsilon
    q=q+epsilon
    return np.sum(np.where(p!=0, p * np.log(p / q), 0))


def infer(lda_filename,data_matrix):
    with open(lda_filename, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)
        inference_probs = lda.transform(data_matrix)
        print(inference_probs, np.sum(inference_probs))
    return inference_probs


def transformation_matrix(lda_filename, data_matrix):
    #data_matrix_normalized = [data / np.sum(data) for data in data_matrix]
    #  data_matrix_normalized = data_matrix / data_matrix.sum(axis=1)[:, np.newaxis]
    inv = np.linalg.pinv(data_matrix)
    print(inv,inv.shape)
    with open(lda_filename, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)
        rhs = lda.components_
        # rhs = rhs / rhs.sum(axis=1)[:, np.newaxis]
        print("components=",rhs, rhs.shape)
    trans = inv@rhs.T
    #trans = np.dot(inv,rhs.T)
    return trans


def get_n_closest_images(query_image, images_latent_space, n, image_ids):
    distances = np.array([kl_divergence(image_vector,query_image) for image_vector in images_latent_space])
    print("distances=",distances)
    top_features_ind = distances.argsort()[:n]
    print("top_features_ind=",top_features_ind)
    top_images = [image_ids[ind] for ind in top_features_ind]
    print('distances=',distances,"topn ids=",top_features_ind," top_n_images=",top_images)
    return top_images
