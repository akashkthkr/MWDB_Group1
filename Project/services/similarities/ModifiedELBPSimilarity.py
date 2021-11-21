import numpy as np

from services.similarities import ChiSquareSimilarity

def get_elbp_similarity(source, features):
    result_dict = {}
    for image in features:
        result = []
        block_result = ChiSquareSimilarity.get_chi_square_similarity(np.array(features[source]["MELBP"]))