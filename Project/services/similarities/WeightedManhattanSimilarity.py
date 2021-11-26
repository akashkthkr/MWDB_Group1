from scipy import spatial
import numpy as np
from services import ScoreCalculator
from utilities import PlotResult

# Manhattan distance for a single image is calculated here
def get_manhattan_similarity(source_image, target_image):
    similarity_dict = {}
    w1, w2, w3 = 0.34, 0.33, 0.33
    val1 = np.abs(np.array(source_image[0:64]) - np.array(target_image[0:64]))
    val2 = np.abs(np.array(source_image[64:128]) - np.array(target_image[64:128]))
    val3 = np.abs(np.array(source_image[128:192]) - np.array(target_image[128:192]))
    return w1 * np.mean(val1) + w2 * np.mean(val2) + w3 * np.mean(val3)
    # return w1 * val1 + w2 * val2 + w3 * val3
    # result = 1 -spatial.distance.cosine(source_image,target_image)
    # return result;

# Manhattan distance for all the images are calculated here
def get_manhattan_similarity_from_merged(source, features, size, dataset, similarity_model):
    result_dict = {}
    for image in features:
        result = get_manhattan_similarity(features[source], features[image])
        result_dict[image] = result
    # sorted_result = ScoreCalculator.get_merged_score(result_dict, size+1)
    sorted_result = sorted(result_dict, key=result_dict.get)
    PlotResult.plot_results(result_dict, sorted_result, size, source, dataset, similarity_model)
    return sorted_result