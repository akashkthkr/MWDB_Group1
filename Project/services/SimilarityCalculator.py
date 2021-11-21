from Project.utilities import MergeFeatures
from Project.services.similarities import WeightedManhattanSimilarity, ChiSquareSimilarity, LNormSimilarity, MergedSimilarity


# Various similarity measures are called through this method and the sorted result is returned
def calculate_similarity(source, features, model, size, dataset):
    if model == "CM":
        merged_features = MergeFeatures.get_merged_feature_list(features)
        result = WeightedManhattanSimilarity.get_manhattan_similarity_from_merged(source, merged_features, size+1,dataset, model)
        # result = LNormSimilarity.get_l_norm_similarity_from_merged(source, features, "Merged Color Moment",2)
    elif model == "ELBP":
        result = ChiSquareSimilarity.get_chi_square_similarity_from_merged(source, features, size+1, dataset, model)
        # result = ChiSquareSimilarity.get_blocked_chi_square_similarity_from_merged(source, features, size)
    elif model == "HOG":
        result = LNormSimilarity.get_l_norm_similarity_from_merged(source, features, "HOG",1, size+1, dataset)
    else:
        result = MergedSimilarity.get_merged_similarities(source, features, size+1, dataset, model)

    return result[0:size+1]
