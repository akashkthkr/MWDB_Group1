from skimage.feature import hog
from skimage.transform import resize


def get_gradient_feature(image):
    row, col = image.shape
    resized_img = resize(image, (128, 64))
    new_row, new_col = resized_img.shape
    vector = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                 multichannel=False, block_norm='L2-Hys')
    # print(vector)
    return vector
