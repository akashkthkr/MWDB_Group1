from Project.services.similarities import WeightedManhattanSimilarity, ChiSquareSimilarity, LNormSimilarity


def get_similarity_score_for_nested_images(dataset, outer_id, inner_id, model_name):
    if model_name == "CM":
        similarity_score = WeightedManhattanSimilarity.get_manhattan_similarity(dataset[outer_id], dataset[inner_id])
    elif model_name == "ELBP":
        similarity_score = ChiSquareSimilarity.get_chi_square_similarity(dataset[outer_id], dataset[inner_id])
    else:
        similarity_score = LNormSimilarity.get_l_norm_similarity(dataset[outer_id], dataset[inner_id], 1)
    return similarity_score


def get_old_similarity_score_for_nested_images(dataset, outer_id, inner_id):
    length = min(len(outer_id), len(inner_id))
    similarity_score = 0
    for id in dataset[outer_id]:
        if id in dataset[inner_id]:
            similarity_score = similarity_score + WeightedManhattanSimilarity.get_manhattan_similarity(dataset[outer_id][id]['Merged Color Moment'], dataset[inner_id][id]['Merged Color Moment'])
    similarity_score = similarity_score/length
    return similarity_score


def get_similarity_matrix(dataset, task_id, model_name):
    similarity_matrix = {}
    if task_id == "3" or task_id =="4":
        for outer_id in dataset:
            nested_similarity_matrix = {}
            for inner_id in dataset:
                nested_similarity_matrix[inner_id] = get_similarity_score_for_nested_images(dataset, outer_id, inner_id, model_name)
            similarity_matrix[outer_id] = nested_similarity_matrix
    elif task_id == "1" or task_id == "2":
        similarity_matrix = dataset

    return similarity_matrix
