import numpy as np

from services.similarities import WeightedManhattanSimilarity, ChiSquareSimilarity, LNormSimilarity
from utilities import MergeFeatures, PlotResult, Normalization
import Constants

# The results of all the feature extraction models for a single image is combined here
def get_weighted_similarity(color_moment, elbp, hog):
    w1, w2, w3 = 0.5, 0.2, 0.3
    modified_hog = (hog - Constants.MIN_VALUE)/(Constants.MAX_VALUE- Constants.MIN_VALUE)
    return (w1 * color_moment + w2 * elbp + w3 * modified_hog)

# The results of all the feature extraction models for all the images are calculated here
def get_merged_similarities(source, features,size,dataset, similarity_model):
    result_dict, final_dict = {},{}
    LNormSimilarity.get_l_norm_similarity_from_merged(source, features, "HOG", 1, size + 1, dataset)
    for image in features:
        merged_features = MergeFeatures.get_merged_feature_list(features)
        color_moment = WeightedManhattanSimilarity.get_manhattan_similarity(np.array(merged_features[source]),
                                                                            np.array(merged_features[image]))
        elbp = ChiSquareSimilarity.get_chi_square_similarity(np.array(features[source]["ELBP"]),
                                                                  np.array(features[image]["ELBP"]))
        hog = LNormSimilarity.get_l_norm_similarity(np.array(features[source]["HOG"]),
                                                                  np.array(features[image]["HOG"]),1)

        result = get_weighted_similarity(color_moment, elbp, hog)
        result_dict[image] = result

    final_dict = Normalization.get_normalized_vector(result_dict)
    sorted_result = sorted(result_dict, key=result_dict.get)
    PlotResult.plot_results(final_dict, sorted_result, size, source, dataset, similarity_model)
    return sorted_result