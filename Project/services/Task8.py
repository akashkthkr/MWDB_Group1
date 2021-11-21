import copy
import math
import networkx as nx
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import json
from numpyencoder import NumpyEncoder


def _is_converge(sim, sim_old, n, eps=1e-4):
    for i in range(n):
        for j in range(n):
            if abs(sim[i, j] - sim_old[i, j]) >= eps:
                return False
    return True


def neighbors(i, n):
    return [ele for ele in range(0, n) if not ele == i]


def nullify_diagonal_elements(matrix, n):
    matrix_length = len(matrix)
    for i in range(0, matrix_length):
        indices_of_top_m_significant = sorted(range(n), key=lambda j: matrix[i][j], reverse=True)[:n]
        matrix[i][i] = 0
        for k in range(0, matrix_length):
            if k not in indices_of_top_m_significant:
                matrix[i][k] = 0


def ascos_plus_plus(matrix, c=0.9):
    matrix_length = len(matrix)
    nodes_count = matrix_length
    subject_ids = [k for k in range(0, matrix_length)]
    subject_id_lookup_tbl = {}
    # aa
    for i, n in enumerate(subject_ids):
        subject_id_lookup_tbl[n] = i
    # aa
    neighbor_ids = [neighbors(i, matrix_length) for i in subject_ids]
    # aa
    del subject_id_lookup_tbl
    final_similarity_matrix = numpy.eye(nodes_count)
    temp_similarity = numpy.zeros(shape=(nodes_count, nodes_count))
    while not _is_converge(final_similarity_matrix, temp_similarity, nodes_count):
        temp_similarity = copy.deepcopy(final_similarity_matrix)
        for i in range(nodes_count):
            for j in range(nodes_count):
                if i == j:
                    continue
                sim_ij = 0.0
                for neighbour_index in neighbor_ids[i]:
                    w_ik = matrix[i][neighbour_index]  # where wik is the weight of edge
                    sim_ij += float(w_ik) * (1 - math.exp(-w_ik)) * temp_similarity[
                        neighbour_index, j]  # sim_ij is similarity score at indices i and j
                w_i = sum([ele for ele in matrix[i]])
                final_similarity_matrix[i, j] = c * sim_ij / w_i if w_i > 0 else 0
    # aa
    return subject_ids, final_similarity_matrix


def return_most_significant_objects(matrix1, n, m):
    plot_graph(matrix1, n)
    matrix = copy.deepcopy(matrix1)
    nullify_diagonal_elements(matrix, n)
    _, results = ascos_plus_plus(matrix)
    final_results = [sum(row_elements) for row_elements in results]
    indices_of_top_m_significant = sorted(range(len(final_results)), key=lambda i: final_results[i], reverse=True)[:m]
    p = {i: final_results[i] for i in indices_of_top_m_significant}
    print(p)
    with open("D:\\Masters\\CSE 515\\Project\\task-8-ascoss.json", 'w') as file:
        jsonString = json.dumps(p, cls=NumpyEncoder)
        file.write(jsonString)


def plot_graph(MATRIX, N):
    MATRIX = normalize_matrix(MATRIX)
    subject_list = get_subjects(numpy.shape(MATRIX)[0])
    df = pd.DataFrame(MATRIX, columns=subject_list, index=subject_list)
    corr = get_corr_matrix(df, subject_list, N)
    G = create_graph(subject_list, corr)
    nx.draw(G, with_labels=True)
    plt.savefig("D:\\Masters\\CSE 515\\Project\\task-8-graph.png")
    plt.show()


def normalize_matrix(MATRIX):
    for outer_index in range(MATRIX.shape[0]):
        for inner_index in range(MATRIX.shape[1]):
            MATRIX[outer_index][inner_index] = 1 - MATRIX[outer_index][inner_index]
    return MATRIX


def get_subjects(shape):
    subjects = [f"{x + 1}" for x in range(shape)]
    return subjects


def get_corr_matrix(dataframe, subjects, N):
    corr = []
    for item in subjects:
        temp = []
        for i in range(len(dataframe[item])):
            if (item != f'{i + 1}'):
                temp.append(
                    [item, f'{i + 1}', dataframe[item][i]]
                )
        temp.sort(key=lambda a: a[2])
        for val in temp[:N]:
            corr.append(val)
    return corr


def create_graph(nodes, matrix):
    x = nx.DiGraph()
    for node in nodes:
        x.add_node(node)
    for edge in matrix:
        x.add_edge(edge[0], edge[1], weight=edge[2])
    nx.set_node_attributes(x, 1, "pagerank")
    return x
