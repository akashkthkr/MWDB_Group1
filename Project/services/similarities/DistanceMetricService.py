from constants import Constants_phase2 as Constants_p2
from services import ResultSubmission
from services.similarities import LNormSimilarity
from utilities import PlotResult


def get_similar_images(path, features, query_image_id, similar_images_size):
    result = {}
    result_dict = {}
    for image_id in features:
        if image_id == query_image_id and not Constants_p2.QUERY_IMAGE_FOUND:
            continue
        result_dict[image_id] = LNormSimilarity.get_l_norm_similarity(features[query_image_id], features[image_id], 2)
    sorted_result = sorted(result_dict, key=result_dict.get)
    if Constants_p2.QUERY_IMAGE_FOUND:
        result = sorted_result[0:similar_images_size + 1]
    else:
        result = sorted_result[0:similar_images_size]
    ResultSubmission.write_result(path, result)
    PlotResult.plot_results(Constants_p2.IMAGE_SET, sorted_result, similar_images_size, query_image_id, result_dict, None)
    return result

def get_image_type_assosciation(path, features, query_image_id):
    result = {}
    result_dict = {}
    for image_id in features:
        if image_id == query_image_id and not Constants_p2.QUERY_IMAGE_FOUND:
            continue
        result_dict[image_id] = LNormSimilarity.get_l_norm_similarity(features[query_image_id], features[image_id], 2)
    sorted_result = sorted(result_dict, key=result_dict.get)
    # ResultSubmission.write_result(path, sorted_result)
    PlotResult.plot_results(Constants_p2.IMAGE_SET, sorted_result, 5, query_image_id, result_dict, None)
    if Constants_p2.QUERY_IMAGE_FOUND:
        result = sorted_result[1]
    else:
        result = sorted_result[0]
    return result

def get_image_subject_assosciation(path, features, query_image_id):
    result = {}
    result_dict = {}
    for image_id in features:
        if image_id == query_image_id and not Constants_p2.QUERY_IMAGE_FOUND:
            continue
        result_dict[image_id] = LNormSimilarity.get_l_norm_similarity(features[query_image_id], features[image_id], 2)
    sorted_result = sorted(result_dict, key=result_dict.get)
    # ResultSubmission.write_result(path, sorted_result)
    PlotResult.plot_results(Constants_p2.IMAGE_SET, sorted_result, 5, query_image_id, result_dict, None)
    if Constants_p2.QUERY_IMAGE_FOUND:
        result = sorted_result[1]
    else:
        result = sorted_result[0]
    return result
