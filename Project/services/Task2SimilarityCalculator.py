from Project.services.similarities import LNormSimilarity


def get_eucledian_distance(dataset):
    result = {}
    count = 1
    for outer_image in dataset:
        outer_dict = {}
        for inner_image in dataset:
            outer_dict[inner_image] = LNormSimilarity.get_l_norm_similarity(dataset[outer_image], dataset[inner_image],
                                                                            2)
        sorted_result = sorted(outer_dict, key=outer_dict.get)
        result[outer_image] = sorted_result
        print(count)
        count = count + 1
    return result


def get_eucledian_distance_compared_to_type_latent(dataset, type_latent_features):
    result = {}
    count = 1
    for outer_image in dataset:
        outer_dict = {}
        for inner_image in type_latent_features:
            outer_dict[inner_image] = LNormSimilarity.get_l_norm_similarity(dataset[outer_image],
                                                                            type_latent_features[inner_image], 2)
        sorted_result = sorted(outer_dict, key=outer_dict.get)
        result[outer_image] = sorted_result
        print(count)
        count = count + 1
    return result


def get_eucledian_distance_compared_to_subject_latent(dataset, subject_latent_features):
    result = {}
    count = 1
    for outer_image in dataset:
        outer_dict = {}
        for inner_image in subject_latent_features:
            outer_dict[inner_image] = LNormSimilarity.get_l_norm_similarity(dataset[outer_image],
                                                                            subject_latent_features[inner_image], 2)
        sorted_result = sorted(outer_dict, key=outer_dict.get)
        result[outer_image] = sorted_result
        print(count)
        count = count + 1
    return result
