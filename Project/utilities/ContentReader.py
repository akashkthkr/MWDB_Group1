import csv

import numpy as np


def get_matrix(file_name):
    transformation_matrix = []
    with open(file_name) as csv_file:
        csv_content = csv.reader(csv_file)
        for lines in csv_content:
            transformation_matrix.append([float(i) for i in lines])
    return np.array(transformation_matrix)


def get_transformation_matrix(file_name):
    transformation_matrix = get_matrix(file_name)
    # return transformation_matrix
    min_value = np.min(transformation_matrix)
    max_value = np.max(transformation_matrix)
    for index in range(transformation_matrix.shape[0]):
        for inner_index in range(transformation_matrix.shape[1]):
            transformation_matrix[index][inner_index] = (transformation_matrix[index][inner_index] - min_value) / (
                        max_value - min_value)
    return transformation_matrix
