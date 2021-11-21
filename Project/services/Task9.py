import json

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from datetime import datetime
from numpyencoder import NumpyEncoder


def get_subjects(shape):
    subjects = [f"{x + 1}" for x in range(shape)]
    return subjects


def get_corr_matrix(dataframe, subjects, N):
    corr = []
    for item in subjects:
        temp = []
        for i in range(len(dataframe[item])):
            if item != f'{i + 1}':
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


def pagerank(G, personalization, m):
    alpha = 0.85
    max_iter = 100
    tol = 1.0e-6

    result = {}
    x = dict.fromkeys(G, 1.0 / G.number_of_nodes())

    s = float(sum(personalization.values()))
    p = dict((k, v / s) for k, v in personalization.items())

    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        calc = 0
        for n in x:
            for nbr in G[n]:
                x[nbr] += alpha * xlast[n] * G[n][nbr]['weight']
            x[n] += calc * p[n] + (1.0 - alpha) * p[n]

        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < (G.number_of_nodes()) * tol:
            x = {key: value for key, value in sorted(
                x.items(), key=lambda item: item[1], reverse=True)}

            for item in x.items():
                if (personalization[item[0]] == 1):
                    continue

                if len(result) < m:
                    result[f"{item[0]}"] = item[1]
                else:
                    break
            return result


def normalize_matrix(MATRIX):
    for outer_index in range(MATRIX.shape[0]):
        for inner_index in range(MATRIX.shape[1]):
            MATRIX[outer_index][inner_index] = 1 - MATRIX[outer_index][inner_index]
    return MATRIX


def get_similar_subjects_using_ppr(MATRIX, N, M, SUBJECT_LIST):
    MATRIX = normalize_matrix(MATRIX)
    subject_list = get_subjects(np.shape(MATRIX)[0])
    df = pd.DataFrame(MATRIX, columns=subject_list, index=subject_list)
    corr = get_corr_matrix(df, subject_list, N)
    G = create_graph(subject_list, corr)
    nx.draw(G, with_labels=True)
    plt.savefig("D:\\Masters\\CSE 515\\Project\\task-9-graph.png")
    plt.show()

    personalization = {k: 0 for k in subject_list}
    for subject in SUBJECT_LIST:
        personalization[subject] = 1

    p = pagerank(G, personalization, M)
    p = {key: value for key, value in sorted(
        p.items(), key=lambda item: item[1], reverse=True)}

    print(p)
    with open("D:\\Masters\\CSE 515\\Project\\task-9-ppr.json", 'w') as file:
        jsonString = json.dumps(p, cls=NumpyEncoder)
        file.write(jsonString)
