from constants import Constants_phase2 as Constants_p2
from utilities import DataBaseService
import os

path = "./inputs/test_imgage_sets_phase2/set1"


def user_prompt():
    os.chdir("../")
    subjects = None
    similarity_matrix = None
    task_8_9_n = None
    task_8_9_m = None
    query_image_path = None
    images_path = None
    image = None
    subject_type_id = None
    dimensionality_reduction_model = None
    similar_image_size = None
    file_name = None
    k = None
    model_name = None
    task_id = input("Enter the task Id:")
    if task_id == "1":
        # k,model_name, images_path, subject_type_id, dimensionality_reduction_model = "20", "CM", "nhkjds", "con", "KMeans"
        k = input("Enter no of latent semantics value, K:")
        model_name = input("From given options, enter model name in same format: CM, ELBP, HOG")
        images_path = input("Enter images Path")
        subject_type_id = input("Enter value of X:")
        dimensionality_reduction_model = input(
            "From given options, enter dimensional reduction techniques: PCA, SVD, LDA, KMeans")
    if task_id == "2":
        # k, model_name, images_path, subject_type_id, dimensionality_reduction_model = "20", "cm", "nhkjds", "con", "KMeans"
        k = input("Enter no of latent semantics value, K:")
        model_name = input("From given options, enter model name in same format: CM, ELBP, HOG")
        images_path = input("Enter images Path")
        subject_type_id = input("Enter value of Y:")
        dimensionality_reduction_model = input(
            "From given options, enter dimensional reduction techniques: PCA, SVD, LDA, KMeans")
    if task_id == "3":
        # k, model_name, images_path, dimensionality_reduction_model = "20", "cm", "nhkjds", "KMeans"
        k = input("Enter no of latent semantics value, K:")
        model_name = input("From given options, enter model name in same format: CM, ELBP, HOG")
        images_path = input("Enter images Path")
        dimensionality_reduction_model = input(
            "From given options, enter dimensional reduction techniques: PCA, SVD, LDA, KMeans")

    if task_id == "4":
        # k, model_name, images_path, dimensionality_reduction_model = "20", "CM", "nhkjds", "SVD"
        k = input("Enter no of latent semantics value, K:")
        model_name = input("From given options, enter model name in same format: CM, ELBP, HOG")
        images_path = input("Enter images Path")
        dimensionality_reduction_model = input(
            "From given options, enter dimensional reduction techniques: PCA, SVD, LDA, KMeans")

    if task_id == "5":
        # images_path, image_available, file_name, image, similar_image_size = "dummy", "Yes", "lda_latent_semantics_task1", "image-con-30-1", "6"
        print("Before Executing " + task_id + ", make sure to execute either of the tasks from (1,2,3,4)")
        images_path = input("Enter images Path")
        image = input("Enter Query Image name:")
        image_available = input("Is query image available in the same database: Yes/No")
        if image_available == "No":
            query_image_path = input("Enter Query image path")
            query_image_path = query_image_path + r"/" + image + ".png"
            Constants_p2.QUERY_IMAGE_PATH = query_image_path
        file_name = input(
            "Enter Latent Features file Name, file name is in the format -> 'Transformation_Matrix_taskId_FeatureName_TransformationName.csv', example:- Transformation_Matrix_1_CM_PCA.csv")
        similar_image_size = input("Enter the value of most similar n images")

    if task_id == "6" or task_id == "7":
        # images_path, image_available, file_name, image = "dummy", "Yes", "lda_latent_semantics_task1", "image-con-30-1"
        print("Before Executing " + task_id + ", make sure to execute either of the tasks from (1,2,3,4)")
        images_path = input("Enter images Path")
        image = input("Enter Query Image name:")
        image_available = input("Is query image available in the same database: Yes/No")
        if image_available == "No":
            query_image_path = input("Enter Query image path")
            query_image_path = query_image_path + r"/" + image + ".png"
            Constants_p2.QUERY_IMAGE_PATH = query_image_path
        file_name = input(
            "Enter Latent Features file Name, file name is in the format -> 'Transformation_Matrix_taskId_FeatureName_TransformationName.csv', example:- Transformation_Matrix_1_CM_PCA.csv")

    if task_id == "8":
        print("Make sure to execute task 4 before executing task 8 for generating similarity matrix")
        # similarity_matrix, task_8_9_n, task_8_9_m = "Subject_Similarity_Matrix.csv", "5", "10"
        similarity_matrix = input("Enter Similarity matrix file name")
        task_8_9_n = input("Enter value of n")
        task_8_9_m = input("Enter value of m")

    if task_id == "9":
        print("Make sure to execute task 4 before executing task 8 for generating similarity matrix")
        # similarity_matrix, task_8_9_n, task_8_9_m, subjects = "Subject_Similarity_Matrix.csv", "4", "4",["27","28","17"]
        similarity_matrix = input("Enter Similarity matrix file name")
        task_8_9_n = input("Enter value of n")
        task_8_9_m = input("Enter value of m")
        subject_id_1 = input("Enter Subject Id 1")
        subject_id_2 = input("Enter Subject Id 2")
        subject_id_3 = input("Enter Subject Id 3")
        subjects = [subject_id_1, subject_id_2, subject_id_3]

    return images_path, task_id, model_name, subject_type_id, k, dimensionality_reduction_model, image, file_name, query_image_path, similar_image_size, similarity_matrix, task_8_9_n, task_8_9_m, subjects


#  Call to database is made here
def fetch_data(path, task_id, model_name, subject_type_id, k, dimensionality_reduction_model, image_id, file_name,
               query_image_path, similar_image_size, similarity_matrix, task_8_9_n, task_8_9_m, subjects):
    DataBaseService.store_data(path, task_id, model_name, subject_type_id, k, dimensionality_reduction_model, image_id,
                               file_name, query_image_path, similar_image_size, similarity_matrix, task_8_9_n,
                               task_8_9_m, subjects)


# Entry point for this project
if __name__ == '__main__':
    # image, model_name, similar_image_size = "image-0", "Color Moment", 4
    # k, image_id, task_id = 6, "con", "3"
    images_path, task_id, model_name, subject_type_id, k, dimensionality_reduction_model, image_id, file_name, query_image_path, similar_image_size, similarity_matrix, task_8_9_n, task_8_9_m, subjects = user_prompt()
    # images_path = Constants_p2.PATH
    dataset = fetch_data(images_path, task_id, model_name, subject_type_id, k, dimensionality_reduction_model, image_id,
                         file_name, query_image_path, similar_image_size, similarity_matrix, task_8_9_n, task_8_9_m,
                         subjects)
    # print(dataset)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
