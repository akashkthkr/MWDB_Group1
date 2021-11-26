# 3 color moments are merged as list in this function
def merge_color_moment(features):
    feature_list = []
    mean_value = features["Mean"]
    standard_values = features["Standard Deviation"]
    skewness = features["Skewness"]
    feature_list.extend(mean_value.values())
    feature_list.extend(standard_values.values())
    feature_list.extend(skewness.values())
    return feature_list


# This function merges various color moments of all the images
def get_merged_feature_list(features):
    merged_features_dict = {}
    for image in features:
        feature_list = merge_color_moment(features[image]["Color Moment"])
        merged_features_dict[image] = feature_list
    return merged_features_dict
