import numpy as np


def get_sorted_result(features_dict):
    summation_dict = {}
    for image in features_dict:
        summation_dict[image] = np.sum(features_dict[image])
    return sorted(summation_dict, key=summation_dict.get, reverse=True)


def get_merged_score(features_dict, nearest_values):
    for index in range(64):
        size = nearest_values
        image_features_dict = {}
        for image in features_dict:
            image_features_dict[image] = features_dict[image][index]
        sorted_image_features_dict = sorted(image_features_dict, key=image_features_dict.get)

        for image in sorted_image_features_dict:
            features_dict[image][index] = size
            size = size - 1

    sorted_result = get_sorted_result(features_dict)
    return sorted_result
