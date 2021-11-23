import json
from numpyencoder import NumpyEncoder
from Phase3.FeatureGenerationUtilities import save_json_file, get_feature_matrices, generate_features_json, \
    save_transformation_matrix, get_folder_name
from Project.services.dimensionalityReduction import Transformation
from Project.services.dimensionalityReduction.ReductionModel import get_features
from Project.services.ImageFetchService import load_dataset_from_folder_old
from constants import Constants_Phase3

feature_extraction_model = ["CM", "ELBP", "HOG"]
dimensionality_reduction_model = ["PCA", "SVD", "KMeans"]
k_values_list = [5, 10, 20, 50, 100]
feature_storage_path = Constants_Phase3.DATASETS_PATH


def generate_features(path):
    for model_name in feature_extraction_model:
        dataset = load_dataset_from_folder_old(path, model_name)
        json_string = json.dumps(dataset, cls=NumpyEncoder)
        file_name = feature_storage_path + model_name + get_folder_name(path) +"_features.json"
        save_json_file(json_string, file_name)
        feature_matrix = get_feature_matrices(dataset)
        for dimensionality_model in dimensionality_reduction_model:
            for k_value in k_values_list:
                reduced_features = get_features(feature_matrix, dimensionality_model, k_value)
                transformation_matrix = Transformation.get_transformation_matrix(dataset, reduced_features)
                reduced_features_json = generate_features_json(dataset, reduced_features)
                reduced_features_json = json.dumps(reduced_features_json, cls=NumpyEncoder)
                features_file_name = feature_storage_path + model_name + "_" + dimensionality_model + "_" + str(k_value) + get_folder_name(path) + "_features.json"
                transformation_matrix_file_name = feature_storage_path + model_name + "_" + dimensionality_model + "_" + str(k_value) + get_folder_name(path) + "_transformation_matrix.csv"
                save_transformation_matrix(transformation_matrix_file_name, transformation_matrix)
                save_json_file(reduced_features_json, features_file_name)
if __name__ == '__main__':
    images_path = input("Enter images folder path:")
    generate_features(images_path)