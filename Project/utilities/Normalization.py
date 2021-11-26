from constants import Constants_phase2 as Constants_p2


# Values of a vector are normalized from 0 to 1
def get_normalized_vector(init_dict):
    result_list = init_dict.values()
    min_value = min(result_list)
    max_value = max(result_list)
    Constants_p2.MIN_VALUE = min_value
    Constants_p2.MAX_VALUE = max_value
    for image in init_dict:
        init_dict[image] = (init_dict[image] - min_value) / (max_value - min_value)
    return init_dict
