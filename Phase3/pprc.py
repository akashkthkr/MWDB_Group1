import json
import time
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from numpyencoder import NumpyEncoder

from constants.Constants_Phase3 import OUTPUTS_PATH


def ppr(sim_graph, images_list, query_images, max_iter=500, alpha=0.85):
    sim_graph = sim_graph.T
    teleport_matrix = np.array([0 if img not in query_images else 1 for img in images_list]).reshape(len(images_list),
                                                                                                     1)
    teleport_matrix = teleport_matrix / len(query_images)
    uq_new = teleport_matrix
    uq_old = np.array((len(images_list), 1))
    iter = 0
    while iter < max_iter and not np.array_equal(uq_new, uq_old):
        uq_old = uq_new.copy()
        uq_new = alpha * np.matmul(sim_graph, uq_old) + \
                 (1 - alpha) * teleport_matrix
        iter += 1
    # print("Iterations: {}".format(iter))
    uq_new = uq_new.ravel()
    # uq_new = uq_new[::-1].argsort(axis=0)
    a = (-uq_new).argsort()
    result = []
    rank = 1
    res_dict = {}
    for i in a:
        res = {"imageId": images_list[i], "score": uq_new[i], "rank": rank}
        res_dict[images_list[i]] = rank
        result.append(res)
        # print("Image: {} Score: {} Rank:{}".format(
        #     images_list[i], uq_new[i], rank))
        rank += 1
    # return result
    return res_dict


def filter_by_X(data):
    result = {}

    for item in data:
        label = item.split('-')[1]
        try:
            result[f"{label}"].append(item)
        except:
            result[f"{label}"] = []
            result[f"{label}"].append(item)

    return result


def filter_by_Y(data):
    result = {}

    for item in data:
        label = item.split('-')[2]
        try:
            result[f"{label}"].append(item)
        except:
            result[f"{label}"] = []
            result[f"{label}"].append(item)

    return result


def filter_by_Z(data):
    result = {}

    for item in data:
        label = item.split('-')[3]
        try:
            result[f"{label}"].append(item)
        except:
            result[f"{label}"] = []
            result[f"{label}"].append(item)

    return result


def classify_by_X(image_lab, image_unlab, feat_lab, feat_unlab):
    image_list = np.concatenate((image_lab, image_unlab))
    features_list = np.concatenate((feat_lab, feat_unlab))

    labelled = filter_by_X(image_lab)

    print("Calculating similarity...")
    cos_sim = cosine_similarity(features_list)
    print("Similarity calculation complete.")

    print("Calculating similarity matrix...")
    sim_graph = np.empty((0, len(cos_sim)))
    for row in cos_sim:
        k_largest = np.argsort(-np.array(row))[1:10 + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)
    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    print("Similarity matrix calculation complete.")

    ppr_dict = {}

    print("Calculating PPR...")
    for i in labelled:
        ppr_dict[f"{i}"] = ppr(sim_graph, image_list, labelled[i])
    print("PPR calculating complete.")

    print("Classification started...")
    ranks = {}
    for img in image_unlab:
        temp = {}
        for i in list(labelled.keys()):
            temp[f"{i}"] = ppr_dict[f"{i}"][img]
        ranks[img] = temp

    for x in ranks:
        temp = ranks[x]
        temp = {key: value for key, value in sorted(
            temp.items(), key=lambda item: item[1])}
        ranks[x] = temp

    with open(OUTPUTS_PATH + "PPR_task1_ranks.json", "w") as file:
        jsonString = json.dumps(ranks)
        file.write(jsonString)

    labels = {}
    for img in ranks:
        temp = list(ranks[img].items())[0][0]
        labels[img] = temp

    # print(labels)
    with open(OUTPUTS_PATH + "PPR_task1_results.json", "w") as file:
        jsonString = json.dumps(labels)
        file.write(jsonString)
    print("Classification complete.")

    pos = {k: 0 for k in list(labelled.keys())}
    miss = {k: 0 for k in list(labelled.keys())}
    tot_pos = 0
    tot_miss = 0
    total = 0

    for i in labels:
        label = i.split("-")[1]
        total += 1
        if (label != labels[i]):
            miss[label] += 1
            pos[labels[i]] += 1
            tot_pos += 1
            tot_miss += 1

    for i in pos:
        temp = str(pos[i]) + "/" + str(total)
        pos[i] = temp

    for i in miss:
        temp = str(miss[i]) + "/" + str(total)
        miss[i] = temp

    with open(OUTPUTS_PATH + "PPR_task1_false_pos.json", "w") as file:
        jsonString = json.dumps(pos, cls=NumpyEncoder)
        file.write(jsonString)

    with open(OUTPUTS_PATH + "PPR_task1_miss_rates.json", "w") as file:
        jsonString = json.dumps(miss, cls=NumpyEncoder)
        file.write(jsonString)

    print("False Positives - ", tot_pos)
    print("Miss Rates - ", tot_miss)
    print("Total images classifies - ", total)


def classify_by_Y(image_lab, image_unlab, feat_lab, feat_unlab):
    image_list = np.concatenate((image_lab, image_unlab))
    features_list = np.concatenate((feat_lab, feat_unlab))

    labelled = filter_by_Y(image_lab)

    print("Calculating similarity...")
    cos_sim = cosine_similarity(features_list)
    print("Similarity calculation complete.")

    print("Calculating similarity matrix...")
    sim_graph = np.empty((0, len(cos_sim)))
    for row in cos_sim:
        k_largest = np.argsort(-np.array(row))[1:10 + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)
    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    print("Similarity matrix calculation complete.")

    ppr_dict = {}

    print("Calculating PPR...")
    for i in labelled:
        ppr_dict[f"{i}"] = ppr(sim_graph, image_list, labelled[i])
    print("PPR calculating complete.")

    print("Classification started...")
    ranks = {}
    for img in image_unlab:
        temp = {}
        for i in list(labelled.keys()):
            temp[f"{i}"] = ppr_dict[f"{i}"][img]
        ranks[img] = temp

    for x in ranks:
        temp = ranks[x]
        temp = {key: value for key, value in sorted(
            temp.items(), key=lambda item: item[1])}
        ranks[x] = temp

    with open(OUTPUTS_PATH + "PPR_task2_ranks.json", "w") as file:
        jsonString = json.dumps(ranks)
        file.write(jsonString)

    labels = {}
    for img in ranks:
        temp = list(ranks[img].items())[0][0]
        labels[img] = temp

    # print(labels)
    with open(OUTPUTS_PATH + "PPR_task2_results.json", "w") as file:
        jsonString = json.dumps(labels)
        file.write(jsonString)
    print("Classification complete.")

    pos = {k: 0 for k in list(labelled.keys())}
    miss = {k: 0 for k in list(labelled.keys())}
    tot_pos = 0
    tot_miss = 0
    total = 0

    for i in labels:
        label = i.split("-")[2]
        total += 1
        if (label != labels[i]):
            miss[label] += 1
            pos[labels[i]] += 1
            tot_pos += 1
            tot_miss += 1

    for i in pos:
        temp = str(pos[i]) + "/" + str(total)
        pos[i] = temp

    for i in miss:
        temp = str(miss[i]) + "/" + str(total)
        miss[i] = temp

    pos = {k: v for k, v in sorted(pos.items(), key=lambda item: int(item[0]))}
    miss = {k: v for k, v in sorted(miss.items(), key=lambda item: int(item[0]))}

    with open(OUTPUTS_PATH + "PPR_task2_false_pos.json", "w") as file:
        jsonString = json.dumps(pos, cls=NumpyEncoder)
        file.write(jsonString)

    with open(OUTPUTS_PATH + "PPR_task2_miss_rates.json", "w") as file:
        jsonString = json.dumps(miss, cls=NumpyEncoder)
        file.write(jsonString)

    print("False Positives - ", tot_pos)
    print("Miss Rates - ", tot_miss)
    print("Total images classifies - ", total)


def classify_by_Z(image_lab, image_unlab, feat_lab, feat_unlab):
    image_list = np.concatenate((image_lab, image_unlab))
    features_list = np.concatenate((feat_lab, feat_unlab))

    labelled = filter_by_Z(image_lab)

    print("Calculating similarity...")
    cos_sim = cosine_similarity(features_list)
    print("Similarity calculation complete.")

    print("Calculating similarity matrix...")
    sim_graph = np.empty((0, len(cos_sim)))
    for row in cos_sim:
        k_largest = np.argsort(-np.array(row))[1:10 + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)
    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    print("Similarity matrix calculation complete.")

    ppr_dict = {}

    print("Calculating PPR...")
    for i in labelled:
        ppr_dict[f"{i}"] = ppr(sim_graph, image_list, labelled[i])
    print("PPR calculating complete.")

    print("Classification started...")
    ranks = {}
    for img in image_unlab:
        temp = {}
        for i in list(labelled.keys()):
            temp[f"{i}"] = ppr_dict[f"{i}"][img]
        ranks[img] = temp

    for x in ranks:
        temp = ranks[x]
        temp = {key: value for key, value in sorted(
            temp.items(), key=lambda item: item[1])}
        ranks[x] = temp

    with open(OUTPUTS_PATH + "PPR_task3_ranks.json", "w") as file:
        jsonString = json.dumps(ranks)
        file.write(jsonString)

    labels = {}
    for img in ranks:
        temp = list(ranks[img].items())[0][0]
        labels[img] = temp

    # print(labels)
    with open(OUTPUTS_PATH + "PPR_task3_results.json", "w") as file:
        jsonString = json.dumps(labels)
        file.write(jsonString)
    print("Classification complete.")

    pos = {k: 0 for k in list(labelled.keys())}
    miss = {k: 0 for k in list(labelled.keys())}
    tot_pos = 0
    tot_miss = 0
    total = 0

    for i in labels:
        label = i.split("-")[3]
        total += 1
        if (label != labels[i]):
            miss[label] += 1
            pos[labels[i]] += 1
            tot_pos += 1
            tot_miss += 1

    for i in pos:
        temp = str(pos[i]) + "/" + str(total)
        pos[i] = temp

    for i in miss:
        temp = str(miss[i]) + "/" + str(total)
        miss[i] = temp

    pos = {k: v for k, v in sorted(pos.items(), key=lambda item: int(item[0]))}
    miss = {k: v for k, v in sorted(miss.items(), key=lambda item: int(item[0]))}

    with open(OUTPUTS_PATH + "PPR_task3_false_pos.json", "w") as file:
        jsonString = json.dumps(pos, cls=NumpyEncoder)
        file.write(jsonString)

    with open(OUTPUTS_PATH + "PPR_task3_miss_rates.json", "w") as file:
        jsonString = json.dumps(miss, cls=NumpyEncoder)
        file.write(jsonString)

    print("False Positives - ", tot_pos)
    print("Miss Rates - ", tot_miss)
    print("Total images classified - ", total)


def classify_using_ppr(task_id, train_features, test_features):
    start = time.time()

    # print(len(train_features))
    # print(len(test_features))

    image_lab = list(train_features.keys())
    image_unlab = list(test_features.keys())

    features_lab = list(train_features.values())
    features_unlab = list(test_features.values())

    # print(type(features_lab))

    if (task_id == '1'):
        classify_by_X(image_lab, image_unlab, features_lab, features_unlab)
    elif (task_id == '2'):
        classify_by_Y(image_lab, image_unlab, features_lab, features_unlab)
    elif (task_id == '3'):
        classify_by_Z(image_lab, image_unlab, features_lab, features_unlab)

    print("Total time taken - " + str(time.time() - start) + " seconds.")
