import numpy as np
from pathlib import Path
from constants import Constants_phase2 as Constants_p2

from Project.Lda.code.lda import LdaClass
from Project.services.similarities import DistanceMetricService
from Project.utilities import FeatureStorage, LDAChecker
from Project.services import ImageFetchService, SimilarityMatrix, Task8, Task9
from Project.utilities import MatrixConstruction
from Project.services.dimensionalityReduction import Transformation, ReductionModel


def construct_image(dataset):
    final_image = np.zeros(shape = (64,64))
    count = len(dataset)
    for index in range(64):
        for inner_index in range(64):
            for image in dataset:
                final_image[index][inner_index] = final_image[index][inner_index] + dataset[image][index][inner_index]

            value  = round(final_image[index][inner_index]/count)
            # if value >= 0.5:
            #     value = 1
            # else:
            #     value = 0
            final_image[index][inner_index] = value
    # final_image = GreyScaleNormalization.get_normalized_image(final_image)
    return final_image
# All the feature extraction functions and similarity measures are call through this method and the result is stored in database through this method
def store_data(images_path, task_id, model_name, subject_type_id, k, dimensionality_reduction_model, image_id, file_name, query_image_path, similar_image_size, similarity_matrix_name, task_8_9_n, task_8_9_m, subjects):
    # dataset = ImageFetchService.load_dataset_for_task1(path, image_id)
    # dataset = datasets.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)
    # image = construct_image(dataset)
    # plt.imshow(image,cmap='gray')
    # plt.show()
    try:
        is_lda = LDAChecker.is_lda(file_name)
        if dimensionality_reduction_model != "LDA" and not is_lda:
            dataset = ImageFetchService.load_dataset_from_folder(images_path, task_id, model_name, subject_type_id, image_id, file_name, query_image_path, similarity_matrix_name)
            final_features_dict = {}
            # FeatureExtractorService.get_features(dataset, task_id)
            # features_dict = FeatureExtractorService.get_features(dataset, task_id)
            # required_matrix = None
            if int(task_id) < 5:
                features_matrix = SimilarityMatrix.get_similarity_matrix(dataset, task_id, model_name)
                calculated_matrix = MatrixConstruction.get_matrix(features_matrix, task_id)
                # calculated_matrix = MatrixConstruction.normalize_matrix(calculated_matrix)
                reduced_data = ReductionModel.get_features(calculated_matrix, dimensionality_reduction_model, int(k))
                transformation_matrix = Transformation.get_transformation_matrix(dataset, reduced_data)
                FeatureStorage.store_features(features_matrix, reduced_data, transformation_matrix, task_id, model_name, dimensionality_reduction_model, calculated_matrix)
            elif int(task_id) == 5:
                similarity_images_set = DistanceMetricService.get_similar_images(images_path, dataset, image_id, int(similar_image_size))
            elif int(task_id) == 6:
                image_assosciation = DistanceMetricService.get_image_type_assosciation(images_path, dataset, image_id)
                print(image_assosciation)
            elif int(task_id) == 7:
                image_assosciation = DistanceMetricService.get_image_subject_assosciation(images_path, dataset, image_id)
                print(image_assosciation)
            elif int(task_id) == 8:
                Task8.return_most_significant_objects(dataset, int(task_8_9_n),int(task_8_9_m))
            elif int(task_id) == 9:
                Task9.get_similar_subjects_using_ppr(dataset, int(task_8_9_n), int(task_8_9_m), subjects)
            # final_features_dict['features']=features_dict
            # TODO
            # calculated_matrix = MatrixConstruction.get_matrix(required_matrix, task_id)
            # U, S, VT =  SVD.get_svd(calculated_matrix, feature_reduction_count)
            # transformation_matrix = Transformation.get_transformation_matrix(dataset, U)
            # transformation_matrix  = ContentReader.get_transformation_matrix()
            # new_dataset = ImageFetchService.load_dataset_from_folder_old(images_path)
            # dataset_new_features = Transformation.get_latent_features(new_dataset, transformation_matrix)
            # type_latent_features = Transformation.get_type_transformed_features(dataset_new_features)
            # subject_latent_features = Transformation.get_subject_transformed_features(dataset_new_features)
            # type_latent_similarity = Task2SimilarityCalculator.get_eucledian_distance_compared_to_type_latent(dataset_new_features, type_latent_features)
            # subject_latent_similarity = Task2SimilarityCalculator.get_eucledian_distance_compared_to_subject_latent(dataset_new_features, subject_latent_features)
            # similarity_images_set = Task2SimilarityCalculator.get_eucledian_distance(dataset_new_features)
            # u, s, vt = scratch_svd.decompose_svd(calculated_matrix, feature_reduction_count)
            # results = SimilarityCalculator.calculate_similarity(source_image, features_dict, model, similar_image_size, dataset)
            # ResultSubmission.write_result(path, results)
            # jsonString = json.dumps(required_matrix, cls=NumpyEncoder)
            # jsonFile = open("ImageData.json","w")
            # jsonFile.write(jsonString)
            # jsonFile.close()

            # jsonString = json.dumps(final_features_dict, cls=NumpyEncoder)
            # jsonFile = open("ImagesFeatures.json","w")
            # jsonFile.write(jsonString)
            # jsonFile.close()
            # numpy.savetxt('SudhanshuDataSet.txt', dataset['images'][0])
        else:
            # TODO for LDA
            import sys
            sys.path.insert(0,'')
            my_file = Path(Constants_p2.LDA_PATH)
            if not my_file.exists():
                from Project.Lda.code import generate_json
                print("LDA Document generated!!")
            if model_name is not None:
                model_name = model_name.lower()
            lda = LdaClass(model_name, Constants_p2.LDA_PATH)
            if(task_id=='1'):
                lda.task1(model_name,int(k),subject_type_id)
                print("LDA Task 1 done")
            elif(task_id=='2'):
                lda.task2(model_name,int(k),subject_type_id)
                print("LDA Task 2 done")
            elif(task_id=='3'):
                dataset = ImageFetchService.load_dataset_from_folder(images_path, task_id, model_name, subject_type_id,
                                                                     image_id, file_name, query_image_path, similarity_matrix_name)
                features_matrix = SimilarityMatrix.get_similarity_matrix(dataset, task_id, model_name)
                calculated_matrix = MatrixConstruction.get_matrix(features_matrix, task_id)
                lda.task3(np.array(calculated_matrix), int(k))
                print("LDA Task 3 done")
            elif(task_id=='4'):
                dataset = ImageFetchService.load_dataset_from_folder(images_path, task_id, model_name, subject_type_id,
                                                                     image_id, file_name, query_image_path, similarity_matrix_name)
                features_matrix = SimilarityMatrix.get_similarity_matrix(dataset, task_id, model_name)
                calculated_matrix = MatrixConstruction.get_matrix(features_matrix, task_id)
                FeatureStorage.store_similariy_matrix(calculated_matrix)
                lda.task4(np.array(calculated_matrix), int(k))
                print("LDA Task 4 done")
            elif(task_id=='5'):
                if query_image_path is None:
                    query_image_path = images_path+"\\"+image_id+".png"
                top_images = lda.task5(file_name,query_image_path, int(similar_image_size))
                print("LDA Task 5 done"," top_images=",top_images)
            elif(task_id=='6'):
                if query_image_path is None:
                    query_image_path = images_path+"\\"+image_id+".png"
                top_types = lda.task6(file_name, query_image_path)
                print("LDA Task 6 done"," associated_type=",top_types[0], "top types=",top_types)
            elif(task_id=='7'):
                if query_image_path is None:
                    query_image_path = images_path + "\\" + image_id+".png"
                top_subjects = lda.task7(file_name, query_image_path)
                print("LDA Task 7 done","associated_subject=",top_subjects[0]," top subjects=",top_subjects)
            else:
                print("invalid task id")
            #LDA.run_LDA(task_id, file_name, k, image_id, query_image_path, images_path,)
    except IOError as error:
        print("Exception Caught", error)
    print("Data Stored")

# def fetch_data():