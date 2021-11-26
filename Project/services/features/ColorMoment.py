from Project.utilities import BlockArrangement
import math


def get_color_moment_features(image):
    modified_image = BlockArrangement.blockshaped(image, 8, 8)
    features = dict()
    features['Mean'] = get_mean(modified_image)
    features['Standard Deviation'] = get_standard_deviation(modified_image, features['Mean'])
    features['Skewness'] = get_skewness(modified_image, features['Mean'])
    return features


def get_mean(image):
    size, row, col = image.shape
    mean_dict = {}
    for i in range(size):
        img_sum = 0
        for j in range(8):
            for k in range(8):
                img_sum += image[i][j][k]

        mean = img_sum / 64
        x = "Block-" + str(i + 1)
        mean_dict[x] = mean
    return mean_dict


def get_standard_deviation(image, mean):
    size, row, col = image.shape
    sd_dict = {}
    for i in range(size):
        sd_sum = 0
        a = "Block-" + str(i + 1)
        for j in range(8):
            for k in range(8):
                value = image[i][j][k] - mean[a]
                sd_sum += value * value

        sd = sd_sum / 64
        x = "Block-" + str(i + 1)
        sd_dict[x] = math.sqrt(sd)
    return sd_dict


def get_skewness(image, mean):
    size, row, col = image.shape
    skewness_dict = {}
    for i in range(size):
        skew_sum = 0
        a = "Block-" + str(i + 1)
        for j in range(8):
            for k in range(8):
                value = image[i][j][k] - mean[a]
                skew_sum += value * value * value

        skewness = skew_sum / 64
        x = "Block-" + str(i + 1)
        sign = -1 if skewness < 0 else 1
        skewness = skewness * sign
        skewness_dict[x] = sign * (skewness ** (1 / 3))
    return skewness_dict
