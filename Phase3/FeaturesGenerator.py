import glob
import json
import csv
import numpy as np
from Phase3.FeatureGenerationUtilities import generate_file_name, get_feature_matrices, generate_features_json
from Project.services.ImageFetchService import load_dataset_from_folder_old, fetch_features_from_image_path
from Project.services.dimensionalityReduction import ReductionModel, Transformation
from constants.Constants_Phase3 import DATASETS_PATH


def get_features(task_id, feature_model, images_path, query_images_path, query_image_id, query_image_path, reduction_required, dimensionality_reduction_model, k):
    if reduction_required == "yes":
        features_file_name, transformation_matrix_name = generate_file_name(feature_model, reduction_required, images_path, dimensionality_reduction_model, k)
        reduced_features_json, transformation_matrix =  get_features_and_transformation_file(features_file_name, transformation_matrix_name)
        if reduced_features_json == None or transformation_matrix == None:
            dataset = load_dataset_from_folder_old(images_path, feature_model)
            feature_matrix = get_feature_matrices(dataset)
            reduced_features = ReductionModel.get_features(feature_matrix, dimensionality_reduction_model, int(k))
            transformation_matrix = Transformation.get_transformation_matrix(dataset, reduced_features)
            reduced_features_json = generate_features_json(dataset, reduced_features)
        if task_id == "1" or task_id == "2" or task_id == "3":
            query_images_dataset = load_dataset_from_folder_old(query_images_path, feature_model)
            query_image_reduced_features = Transformation.get_latent_features(query_images_dataset,
                                                                              transformation_matrix)
        else:
            id, img, query_image_features = fetch_features_from_image_path(query_image_path, feature_model)
            query_image_reduced_features = {query_image_id: query_image_features}
            query_image_reduced_features = Transformation.get_latent_features(query_image_reduced_features, transformation_matrix)
        return reduced_features_json, query_image_reduced_features
    else:
        file_name = generate_file_name(feature_model, reduction_required, images_path,
                           dimensionality_reduction_model, k)
        dataset = get_original_features_file(file_name)
        if dataset == None:
            dataset = load_dataset_from_folder_old(images_path, feature_model)
        if task_id == "1" or task_id == "2" or task_id == "3":
            query_images_dataset = load_dataset_from_folder_old(query_images_path, feature_model)
            return dataset, query_images_dataset
        else:
            # In case we only want our query image
            id, img, query_images_dataset = fetch_features_from_image_path(query_image_path, feature_model)
            return dataset, {query_image_id: query_images_dataset}

def get_original_features_file(features_file_name):
    features = None
    features_dataset = glob.glob(DATASETS_PATH + "*.json")
    for file_name in features_dataset:
        if(features_file_name in file_name):
            with open(file_name) as json_file:
                features = json.load(json_file)
    return features


def get_features_and_transformation_file(features_file_name, transformation_matrix_name):
    reduced_features, transformation_matrix = None, None
    features_dataset = glob.glob(DATASETS_PATH+"*.json")
    transformation_matrix_dataset = glob.glob(DATASETS_PATH+"*.csv")

    for file_name in features_dataset:
        if(features_file_name in file_name):
            with open(file_name) as json_file:
                reduced_features = json.load(json_file)

    for file_name in transformation_matrix_dataset:
        if (transformation_matrix_name in file_name):
            with open(file_name) as csv_file:
                csv_content = csv.reader(csv_file)
                transformation_matrix = []
                for lines in csv_content:
                    transformation_matrix.append([float(i) for i in lines])

    return reduced_features, transformation_matrix