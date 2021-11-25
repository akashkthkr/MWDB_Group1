from Phase3.ExecutionService import execute_tasks
from Phase3.FeaturesGenerator import get_features
from constants.Constants_Phase3 import PATH_IDENTIFIER


def user_prompt():
    dimensionality_reduction_model, k, query_image_id, query_images_path, query_image_path, classifier = None, None, None, None, None, None
    task_id = input("Enter task number from 1 to 8")
    if task_id == "8":
        return task_id, None, None, None, None, None, None, None, None, None
    feature_model = input("From given options, enter model name in same format: CM, ELBP, HOG")
    images_path = input("Enter Images path")
    if task_id == "1" or task_id == "2" or task_id == "3":
        query_images_path = input("Enter query images path")
        classifier = input("From given options, enter classifier name in same format: SVM, DTC(Decision-Tree), PPR")
    else:
        query_image_id = input("Enter Query Image name:")
        query_image_path = input("Enter Query image path")
        query_image_path = query_image_path + PATH_IDENTIFIER +query_image_id+".png"

    reduction_required = input("Do you want to apply dimensionality reduction technique on extracted features: yes/no")
    if reduction_required == "yes":
        dimensionality_reduction_model = input("From given options, enter dimensional reduction techniques: PCA, SVD, LDA, KMeans")
        k = input("Enter no of latent semantics value, K:")

    return task_id, classifier, feature_model, images_path, query_images_path, query_image_id, query_image_path, reduction_required, dimensionality_reduction_model, k


if __name__ == '__main__':
    task_id, classifier, feature_model, images_path, query_images_path, query_image_id, query_image_path, reduction_required, dimensionality_reduction_model, k = user_prompt()
    # generate_features(images_path)
    train_features, test_features = None, None
    if task_id != "8":
        train_features, test_features = get_features(task_id, feature_model, images_path, query_images_path, query_image_id, query_image_path, reduction_required, dimensionality_reduction_model, k)
    execute_tasks(task_id, train_features, test_features, classifier)