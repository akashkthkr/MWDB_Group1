import json
from numpyencoder import NumpyEncoder
import numpy as np

from Project.utilities import MatrixConstruction


def store_similariy_matrix(subject_similarity_matrix):
    subject_similarity_matrix = MatrixConstruction.normalize_matrix(subject_similarity_matrix)
    np.savetxt("Subject_similarity_Matrix.csv", subject_similarity_matrix, delimiter=",")


def generate_file_name(task_id, model_name, dimensionlaity_reduction_model):
    suffix = "_" + task_id + "_" + model_name + "_" + dimensionlaity_reduction_model
    return "Latent_Features" + suffix + ".json", "Transformation_Matrix" + suffix + ".csv"


def store_features(features_matrix, reduced_data, transformation_matrix, task_id, model_name,
                   dimensionality_reduction_model, subject_similarity_matrix):
    latent_features_file_name, transformation_matrix_file_name = generate_file_name(task_id, model_name,
                                                                                    dimensionality_reduction_model)
    features_json = {}
    index = 0
    for image_id in features_matrix:
        features_json[image_id] = reduced_data[index]
        index = index + 1

    jsonString = json.dumps(features_json, cls=NumpyEncoder)
    jsonFile = open(latent_features_file_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    np.savetxt(transformation_matrix_file_name, transformation_matrix, delimiter=",")

    if task_id == "4":
        store_similariy_matrix(subject_similarity_matrix)
