from termcolor import colored
import vafiles
from pathlib import Path
import numpy as np
from skimage.io import imread

class Input:
    transformation_matrix = None
    def hard_coded_input(self):
        vec = np.array([[1, 3], [2, 3], [4, 10], [13, 6], [18, 1]])
        image_ids = ['0', '1', '2', '3', '4']
        return vec,image_ids

    def hard_coded_query(self):
        return [2,4]

    def create_dataset_files(self, feature_type, task_num, reduction_technique, num_latent_semantics):
        print("create_dataset_files")
        pass
    def get_transformation_matrix_filename(self, feature_type, task_number, reduction_techinque, num_latent_semantics):
        return "./datasets/Transformation_Matrix_"+str(task_number)+"_"+feature_type+"_"+\
               reduction_techinque+"_"+str(num_latent_semantics)+".csv"

    def get_transformed_dataset_filename(self,feature_type, task_number, reduction_techinque, num_latent_semantics):
        return "./datasets/Latent_Features_" + str(
            task_number) + "_" + feature_type + "_" + reduction_techinque + "_" + \
               str(num_latent_semantics)+".json"

    def get_features_and_ids_dim_reduced(self, feature_type, task_num, reduction_technique, num_latent_semantics):
        print('get_features_and_ids_dim_reduced')
        trans_matrix_filename = self.get_transformation_matrix_filename(feature_type,task_num,reduction_technique,num_latent_semantics)
        transformed_dataset_filename = self.get_transformed_dataset_filename(feature_type,task_num,reduction_technique,num_latent_semantics)
        print("trans_matrix_filename",trans_matrix_filename)
        print('transformed_dataset_filename',transformed_dataset_filename)
        trans_matrix_exists = Path(trans_matrix_filename).is_file()
        transformed_dataset_exists = Path(transformed_dataset_filename).is_file()
        if trans_matrix_exists==False or transformed_dataset_exists==False:
            print(colored("Transformed dataset not found for given parameters, Creating new latent semantics files",'red'))
            self.create_dataset_files(feature_type,task_num,reduction_technique,num_latent_semantics)
        else:
            print(colored("Transformed dataset and transformation matrix found for given parameters",'red'))
        trans_matrix_exists = Path(trans_matrix_filename).is_file()
        transformed_dataset_exists = Path(transformed_dataset_filename).is_file()
        assert trans_matrix_exists and transformed_dataset_exists
        # TODO read_files_here
        # self.transformation_matrix = trans_matrix_from_file
        return self.hard_coded_input()


    def get_features_and_ids_non_reduced(self, feature_type):
        print('get_features_and_ids_non_reduced')
        # TODO read_database_here for images and features
        return self.hard_coded_input()

    def get_query_features(self, query_image, feature_type, is_reduced):
        print('get_query_features')
        # TODO extract features of type feature_type for query_image
        if is_reduced:
            pass
            # read transformation_matrix_file
            # multiple
        return self.hard_coded_query()
feedback_algorithm = input("Relevance feedback (DT/SVM)? ")
query_image_path = input("Path to query image: ")
query_image = imread(query_image_path) / 255
print(colored("query_image = "+str(query_image),'blue'))
t = int(input("Number of nearest neighbors required to query image? "))
dimensionality_reduction_required = input("Dimensionality reduction required ? (Y/N):")
feature_type = input("Enter feature type required (CM/ELBP/HOG):")

image_vectors = None
image_ids = None
query_vector = None
if dimensionality_reduction_required == 'Y':
    task_number_phase2 = int(input("Enter task number from phase2 for dimensionality reduced vectors required (1/2/3/4):"))
    dimensionality_reduction_technique = input("PCA/SVD/LDA/KMeans: ")
    k = int(input("Enter number of latent semantics k:"))
    input_obj = Input()
    image_vectors, image_ids = input_obj.get_features_and_ids_dim_reduced(feature_type,task_number_phase2,dimensionality_reduction_technique,k)
    query_vector = input_obj.get_query_features(query_image,feature_type,True)
elif dimensionality_reduction_required == 'N':
    input_obj = Input()
    image_vectors, image_ids = input_obj.get_features_and_ids_non_reduced(feature_type)
    query_vector = input_obj.get_query_features(query_image,feature_type, False)
else:
    print(colored("Invalid value required for Dimensionality reduction required",'red'))
    exit(1)

print("Image-vectors and Image-Ids::",image_vectors,image_ids)
knn_indexing_algorithm = input("VA or LSH based classifier for knn algorithm? ")
knn_results = []
if(knn_indexing_algorithm=="VA"):
    b = int(input("Number of bits per dimension for VA:"))
    vafiles.create_and_save_va_file(image_vectors,b,image_ids,"./outputs")
    # TODO change p_in_lp as per the feature model (HOG/ELBP/CM)
    knn_results = vafiles.va_search('./outputs',query_vector,t,'va_ssa',p_in_lp=1)
elif(knn_indexing_algorithm=="LSH"):
    L = int(input("Number of layers for LSH:"))
    k = int(input("Number of hashes per layer for LSH:"))
    # TODO Add LSH logic here
    knn_results = None
else:
    print(colored("Invalid knn algorithm entered...please check - One of VA or LSH required",'red'))
    exit(1)

# TODO relevance feedback for knn_results

if feedback_algorithm == 'SVM':
    pass
elif feedback_algorithm == 'DT':
    pass
else:
    print(colored("Invalid feedback algorithm entered...please check - One of DT or SVM required",'red'))
    exit(1)