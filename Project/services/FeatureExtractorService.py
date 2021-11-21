from Project.services.features import ColorMoment, LocalBinaryPattern, HistogramGradient, ModifiedLocalBinaryPattern
from Project.utilities import MergeFeatures


# All the features results are clubbed inside this method
def get_features_for_tasks_1_2(dataset):
    features_dict = dataset
    for image in dataset:
        inner_feature_dict = {}
        color_moment_feature = ColorMoment.get_color_moment_features(dataset[image])
        merged_color_moment = MergeFeatures.merge_color_moment(color_moment_feature)
        local_binary_pattern = LocalBinaryPattern.get_local_binary_pattern(dataset[image])
        gradient = HistogramGradient.get_gradient_feature(dataset[image])
        inner_feature_dict['Color Moment'] = color_moment_feature
        inner_feature_dict['ELBP'] = local_binary_pattern
        inner_feature_dict['HOG'] = gradient
        inner_feature_dict['Merged Color Moment'] = merged_color_moment
        features_dict[image] = inner_feature_dict
    return features_dict


def get_features_for_tasks_3_4(dataset):
    features_dict = dataset
    for outer_id in dataset:
        for inner_id in dataset[outer_id]:
            inner_feature_dict = {}
            color_moment_feature = ColorMoment.get_color_moment_features(dataset[outer_id][inner_id])
            merged_color_moment = MergeFeatures.merge_color_moment(color_moment_feature)
            local_binary_pattern = LocalBinaryPattern.get_local_binary_pattern(dataset[outer_id][inner_id])
            gradient = HistogramGradient.get_gradient_feature(dataset[outer_id][inner_id])
            inner_feature_dict['Color Moment'] = color_moment_feature
            inner_feature_dict['ELBP'] = local_binary_pattern
            inner_feature_dict['HOG'] = gradient
            inner_feature_dict['Merged Color Moment'] = merged_color_moment
            features_dict[outer_id][inner_id] = inner_feature_dict
    return features_dict


def get_features(dataset, task_id):
    if task_id == "1" or task_id == "2":
        get_features_for_tasks_1_2(dataset)
    else:
        get_features_for_tasks_3_4(dataset)


def get_old_features(dataset):
    features_dict = {}
    for image in dataset:
        inner_feature_dict = {}
        # a = 'image'+str(i+1)
        color_moment_feature = ColorMoment.get_color_moment_features(dataset[image])
        merged_color_moment = MergeFeatures.merge_color_moment(color_moment_feature)
        local_binary_pattern = LocalBinaryPattern.get_local_binary_pattern(dataset[image])
        gradient = HistogramGradient.get_gradient_feature(dataset[image])
        # dict[image] = images[i]
        inner_feature_dict['Color Moment'] = color_moment_feature
        inner_feature_dict['ELBP'] = local_binary_pattern
        inner_feature_dict['HOG'] = gradient
        # inner_feature_dict['MELBP'] = modified_local_binary_pattern
        features_dict[image] = inner_feature_dict
    return features_dict
