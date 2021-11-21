import numpy as np
import math
from utilities import Normalization, PlotResult

# Lnorm similarity measure for single image is calculated here
def get_l_norm_similarity(vector1, vector2, li):
    assert vector1.shape == vector2.shape and vector1.ndim == 1 and li >= 1
    sm = np.sum(np.power(np.abs(vector1-vector2),li))
    return math.pow(sm, 1/float(li))

# L norm similarity measures for all the images are calculated here
def get_l_norm_similarity_from_merged(source, features, similarity_model, li, size, dataset):

    result_dict, final_dict = {},{}
    for image in features:
        result = get_l_norm_similarity(np.array(features[source][similarity_model]), np.array(features[image][similarity_model]), li)
        result_dict[image] = result
    final_dict = Normalization.get_normalized_vector(result_dict)
    sorted_result = sorted(final_dict, key=result_dict.get)
    PlotResult.plot_results(final_dict, sorted_result, size, source, dataset, similarity_model)
    return sorted_result
