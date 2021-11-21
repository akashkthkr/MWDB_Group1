import numpy as np

# Grey scale image having value greater than 0 is normalized here
def get_normalized_image(image):
    row, col = image.shape
    max_value = np.max(image)
    min_value = np.min(image)
    found = True
    # for i in range(row):
    #     for j in range(col):
    #         if image[i][j] > 1:
    #             found = True
    #             break
    #     if found:
    #         break
    if found:
        for i in range(row):
            for j in range(col):
                image[i][j] = (image[i][j]-min_value)/(max_value - min_value)
    return image