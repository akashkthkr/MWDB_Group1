import numpy as np
from Project.utilities import MatrixConstruction
from Project.services import ImageFetchService


def get_transformation_matrix(original_data, transformed_data):
    original_matrix = np.array(MatrixConstruction.get_matrices_for_task_1_2(original_data))
    original_matrix_inverse = np.linalg.pinv(original_matrix)
    transformation_matrix = original_matrix_inverse.dot(transformed_data)
    return transformation_matrix


def get_latent_features(original_data, transformed_matrix):
    for image_id in original_data:
        original_data[image_id] = np.array(original_data[image_id]).dot(transformed_matrix)
    return original_data


def get_type_transformed_features(dataset):
    result = {}
    type_latent_features = {}
    for image_id in dataset:
        subject_id, type_id, second_id = ImageFetchService.extract_subject_id_image_type_and_second_id(image_id)
        if type_id in result:
            result[type_id].append(dataset[image_id])
        else:
            result[type_id] = []
            result[type_id].append(dataset[image_id])

    for image_id in result:
        outer_list = None
        for inner_list in result[image_id]:
            if outer_list is None:
                outer_list = inner_list
            else:
                outer_list = outer_list + inner_list
        outer_list = outer_list / (len(result[image_id]))
        type_latent_features[image_id] = outer_list
    return type_latent_features


def get_subject_transformed_features(dataset):
    result = {}
    subject_latent_features = {}
    for image_id in dataset:
        subject_id, type_id, second_id = ImageFetchService.extract_subject_id_image_type_and_second_id(image_id)
        if subject_id in result:
            result[subject_id].append(dataset[image_id])
        else:
            result[subject_id] = []
            result[subject_id].append(dataset[image_id])

    for image_id in result:
        outer_list = None
        for inner_list in result[image_id]:
            if outer_list is None:
                outer_list = inner_list
            else:
                outer_list = outer_list + inner_list
        outer_list = outer_list / (len(result[image_id]))
        subject_latent_features[image_id] = outer_list
    return subject_latent_features
