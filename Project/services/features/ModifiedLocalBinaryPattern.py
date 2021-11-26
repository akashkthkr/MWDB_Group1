from Project.utilities import BlockArrangement
from Project.services.features import LocalBinaryPattern


def get_blocked_local_binary_pattern(image):
    modified_image = BlockArrangement.blockshaped(image, 8, 8)
    blocked_list = []
    size, row, col = modified_image.shape
    for block in range(size):
        feature_value = LocalBinaryPattern.get_local_binary_pattern(modified_image[block])
        blocked_list.append(feature_value)
    return blocked_list
