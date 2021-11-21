from scipy.stats import chisquare
import numpy as np
from services import ScoreCalculator
from utilities import PlotResult

# Chi-square similarity measure for single image is calculated here
def get_chi_square_similarity(vector1, vector2):
    for i in range(len(vector1)):
        if vector1[i]==0:
            vector1[i] = np.mean(vector1)
    for i in range(len(vector2)):
        if vector2[i] == 0:
            vector2[i] = np.mean(vector2)

    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                      for (a, b) in zip(vector1, vector2) if not (a==0 and b==0)])
    # chi = chisquare(vector1, vector2)
    return chi
    # return chisquare(vector1, vector2)

# Chi-square similarity measures for all the images are calculated here
def get_chi_square_similarity_from_merged(source, features, size, dataset, similarity_model):

    result_dict = {}
    for image in features:
        result = get_chi_square_similarity(np.array(features[source]["ELBP"]),np.array(features[image]["ELBP"]))
        result_dict[image] = result
    sorted_result = sorted(result_dict, key=result_dict.get)
    PlotResult.plot_results(result_dict, sorted_result, size, source, dataset, similarity_model)
    return sorted_result

def get_blocked_chi_square_similarity_from_merged(source, features, size, dataset, similarity_model):
    result_dict = {}
    for image in features:
        result_list = []
        for index in range(len(features[image]["MELBP"])):
            result = get_chi_square_similarity(np.array(features[source]["MELBP"][index]), np.array(features[image]["MELBP"][index]))
            result_list.append(result)
        result_dict[image] = result_list
    sorted_result = ScoreCalculator.get_merged_score(result_dict, size + 1)
    PlotResult.plot_results(result_dict, sorted_result, size, source, dataset, similarity_model)
    return sorted_result