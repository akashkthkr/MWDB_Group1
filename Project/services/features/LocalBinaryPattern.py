from skimage import feature
import numpy as np
import cv2


def get_local_binary_pattern(image):
    numPoints = 24
    radius = 3

    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    eps = 1e-7
    hist /= (hist.sum() + eps)
    return hist
