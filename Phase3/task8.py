from termcolor import colored
from Phase3.vafiles import *
from pathlib import Path


class Input:
    transformation_matrix = None

    def hard_coded_input(self):
        vec = np.array([[1, 3], [2, 3], [4, 10], [13, 6], [18, 1]])
        image_ids = ['0', '1', '2', '3', '4']
        return vec, image_ids

    def create_dataset_files(self, feature_type, task_num, reduction_technique, num_latent_semantics):
        pass

    def get_transformation_matrix_filename(self, feature_type, task_number, reduction_techinque, num_latent_semantics):
        return "./datasets/Transformation_Matrix_" + str(
            task_number) + "_" + feature_type + "_" + reduction_techinque + "_" + str(num_latent_semantics)

    def get_transformed_dataset_filename(self, feature_type, task_number, reduction_techinque, num_latent_semantics):
        return "./datasets/Latent_Features_" + str(
            task_number) + "_" + feature_type + "_" + reduction_techinque + "_" + str(num_latent_semantics)

    def get_features_and_ids_dim_reduced(self, feature_type, task_num, reduction_technique, num_latent_semantics):
        trans_matrix_filename = self.get_transformation_matrix_filename(feature_type, task_num, reduction_technique,
                                                                        num_latent_semantics)
        transformed_dataset_filename = self.get_transformed_dataset_filename(feature_type, task_num,
                                                                             reduction_technique, num_latent_semantics)
        trans_matrix_exists = Path(trans_matrix_filename).is_file()
        transformed_dataset_exists = Path(transformed_dataset_filename).is_file()
        if trans_matrix_exists == False or transformed_dataset_exists == False:
            print(colored("Transformed dataset not found for given parameters, Creating new latent semantics files",
                          'red'))
            self.create_dataset_files(feature_type, task_num, reduction_technique, num_latent_semantics)
        trans_matrix_exists = Path(trans_matrix_filename).is_file()
        transformed_dataset_exists = Path(transformed_dataset_filename).is_file()
        assert trans_matrix_exists and transformed_dataset_exists

        # TODO read_files_here
        # self.transformation_matrix = trans_matrix_from_file
        return self.hard_coded_input()

    def get_features_and_ids_non_reduced(self, feature_type):
        # TODO read_database_here
        return self.hard_coded_input()


feedback_algorithm = input("Relevance feedback (DT/SVM)? ")
query_image_path = input("Path to query image: ")
t = int(input("Number of nearest neighbors required to query image? "))
dimensionality_reduction_required = int(input("Dimensionality reduction required ? (Y/N):"))
feature_type = input("Enter feature type required (CM/ELBP/HOG):")

image_vectors = None
image_ids = None
if dimensionality_reduction_required == 'Y':
    task_number_phase2 = int(
        input("Enter task number from phase2 for dimensionality reduced vectors required (1/2/3/4):"))
    dimensionality_reduction_technique = input("PCA/SVD/LDA/KMEANS: ")
    k = int(input("Enter number of latent semantics k:"))
    input_obj = Input()
    image_vectors, image_ids = input_obj.get_features_and_ids_non_reduced(feature_type)
elif dimensionality_reduction_required == 'N':
    input_obj = Input()
    image_vectors, image_ids = input_obj.get_features_and_ids_non_reduced(feature_type)
else:
    print(colored("Invalid value required for Dimensionality reduction required", 'red'))
    exit(1)
knn_indexing_algorithm = input("VA or LSH based classifier for knn algorithm? ")

knn_results = []
if knn_indexing_algorithm == "VA":
    b = int(input("Number of bits per dimension for VA:"))
    create_and_save_va_file(image_vectors, b, image_ids, "./outputs")
    pass
elif knn_indexing_algorithm == "LSH":
    L = int(input("Number of layers for LSH:"))
    k = int(input("Number of hashes per layer for LSH:"))
else:
    print(colored("Invalid knn algorithm entered...please check - One of VA or LSH required", 'red'))
    exit(1)

if feedback_algorithm == 'SVM':
    pass
elif feedback_algorithm == 'DT':
    pass
else:
    print(colored("Invalid feedback algorithm entered...please check - One of DT or SVM required", 'red'))
    exit(1)
