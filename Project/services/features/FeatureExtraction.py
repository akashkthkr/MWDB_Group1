from Project.services.features import ColorMoment, LocalBinaryPattern, HistogramGradient
from Project.utilities import MergeFeatures


def get_features(image, model_name):
    if model_name == "CM":
        color_moment_feature = ColorMoment.get_color_moment_features(image)
        features = MergeFeatures.merge_color_moment(color_moment_feature)
    elif model_name == "ELBP":
        features = LocalBinaryPattern.get_local_binary_pattern(image)
    else:
        features = HistogramGradient.get_gradient_feature(image)

    return features
