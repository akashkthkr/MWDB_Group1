import matplotlib.pyplot as plt

from constants import Constants_phase2 as Constants_p2

from Project.services import ImageFetchService


def plot_results(data, sorted_result, k, source, final_dict, model):
    fig, ax = plt.subplots(nrows=1, ncols=k + 1)
    source_image = None
    if Constants_p2.QUERY_IMAGE_FOUND:
        source_image = data[source]
    else:
        source_image = ImageFetchService.fetch_image(Constants_p2.QUERY_IMAGE_PATH)
    ax[0].imshow(source_image, cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel(str(source))
    ax[0].set_title("Source Image:")

    if model is None:
        fig.suptitle("Images similar to " + source + " :")
    else:
        fig.suptitle("Images similar to " + source +
                     " based on " + model + " model.")

    for index in range(k):
        ax[index + 1].imshow(data[sorted_result[index]], cmap='gray')
        ax[index + 1].set_xticks([])
        ax[index + 1].set_yticks([])
        ax[index + 1].set_xlabel(str(sorted_result[index]) +
                                 '\n' + str(round(final_dict[sorted_result[index]], 3)))

    plt.show()