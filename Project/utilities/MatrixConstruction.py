import numpy as np


def get_matrices_for_task_3_4(dataset):
    outer_list = []
    for outer_id in dataset:
        inner_list = []
        for inner_id in dataset[outer_id]:
            inner_list.append(dataset[outer_id][inner_id])
        outer_list.append(inner_list)
    return outer_list


def get_matrices_for_task_1_2(dataset):
    outer_list = []
    for outer_id in dataset:
        outer_list.append(dataset[outer_id])
    return outer_list


def get_matrix(dataset, task_id):
    if task_id == "3" or task_id == "4":
        return get_matrices_for_task_3_4(dataset)
    else:
        return get_matrices_for_task_1_2(dataset)


def normalize_matrix(matrix):
    matrix = np.array(matrix)
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    for index in range(matrix.shape[0]):
        for inner_index in range(matrix.shape[1]):
            matrix[index][inner_index] = (matrix[index][inner_index] - min_value) / (
                    max_value - min_value)
    return matrix
