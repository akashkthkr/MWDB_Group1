import numpy as np

from constants.Constants_Phase3 import PATH_IDENTIFIER


def save_json_file(json_string, file_name):
    jsonFile = open(file_name, "w")
    jsonFile.write(json_string)
    jsonFile.close()

def save_transformation_matrix(transformation_matrix_file_name, transformation_matrix):
    np.savetxt(transformation_matrix_file_name, transformation_matrix, delimiter=",")

def get_feature_matrices(dataset):
    outer_list = []
    for outer_id in dataset:
        outer_list.append(dataset[outer_id])
    return outer_list

def generate_features_json(dataset, reduced_features):
    features_json = {}
    index = 0
    for image_id in dataset:
        float_list = [float(value) for value in reduced_features[index]]
        float_list.sort(reverse=True)
        features_json[image_id] = float_list
        index = index + 1
    return features_json

def generate_file_name(feature_model, reduction_required, images_path, dimensionality_reduction_model=None, k=None):
    folder_name = get_folder_name(images_path)
    if reduction_required == "yes":
        features_file_name = feature_model + "_" + dimensionality_reduction_model + "_" + k + folder_name + "_features.json"
        transformation_matrix_name = feature_model + "_" + dimensionality_reduction_model + "_" + k + folder_name + "_transformation_matrix.csv"
        return features_file_name, transformation_matrix_name
    else:
        file_name = feature_model + folder_name + "_features.json"
        return file_name


def get_transformed_latent_features(original_data, transformed_matrix):
    for image_id in original_data:
        original_data[image_id] = np.array(original_data[image_id]).dot(transformed_matrix)
    return original_data

def get_folder_name(path):
    index = path.rfind(PATH_IDENTIFIER) + 1
    return "_"+path[index:]