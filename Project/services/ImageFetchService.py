import numpy as np
from sklearn import datasets
from skimage.io import imread_collection, imread
from skimage.color import rgb2gray
import glob
import os
import ntpath
from constants import Constants_phase2 as Constants_p2
from Project.services.dimensionalityReduction import Transformation
from Project.services.features import ColorMoment, LocalBinaryPattern, HistogramGradient, ModifiedLocalBinaryPattern, \
    FeatureExtraction
from Project.utilities import GreyScaleNormalization, MergeFeatures, ContentReader
from matplotlib import pyplot as plt


def plot_images(dataset):
    for id in dataset:
        if isinstance(dataset[id], list):
            plt.imshow(dataset[id], cmap='gray')
            plt.show()
            print("Done")
        else:
            for inner_id in dataset[id]:
                plt.imshow(dataset[id][inner_id], cmap='gray')
                plt.show()
                print("Done")


def extract_features_for_task_1_2(dataset, model_name):
    for id in dataset:
        final_features = None
        for img_path in dataset[id]:
            img = imread(img_path, as_gray=True)
            new_img = np.array(img).astype(np.float)
            GreyScaleNormalization.get_normalized_image(new_img)
            features = FeatureExtraction.get_features(new_img, model_name)
            # color_moment_feature = ColorMoment.get_color_moment_features(new_img)
            # merged_color_moment = MergeFeatures.merge_color_moment(color_moment_feature)
            if final_features is None:
                final_features = np.array(features)
            else:
                final_features = final_features + np.array(features)
        final_features = final_features / len(dataset[id])
        dataset[id] = final_features
    return dataset


def construct_dataset_for_task_1_2(dataset):
    for id in dataset:
        inner_img = np.zeros(shape=(64, 64))
        for img_path in dataset[id]:
            img = imread(img_path, as_gray=True)
            new_img = np.array(img).astype(np.float)
            inner_img = inner_img + new_img
        inner_img = inner_img / len(dataset[id])
        GreyScaleNormalization.get_normalized_image(inner_img)
        dataset[id] = inner_img
    # plot_images(dataset)
    return dataset


def construct_dataset_for_task_3_4(dataset):
    for id in dataset:
        for inner_id in dataset[id]:
            inner_img = np.zeros(shape=(64, 64))
            for img_path in dataset[id][inner_id]:
                img = imread(img_path, as_gray=True)
                new_img = np.array(img).astype(np.float)
                inner_img = inner_img + new_img
            inner_img = inner_img / len(dataset[id])
            GreyScaleNormalization.get_normalized_image(inner_img)
            dataset[id][inner_id] = inner_img
    # plot_images(dataset)
    return dataset


def extract_features_for_task_3_4(dataset, model_name):
    for id in dataset:
        final_features = None
        inner_img = np.zeros(shape=(64, 64))
        count = 0
        for inner_id in dataset[id]:
            for img_path in dataset[id][inner_id]:
                img = imread(img_path, as_gray=True)
                new_img = np.array(img).astype(np.float)
                GreyScaleNormalization.get_normalized_image(new_img)
                features = FeatureExtraction.get_features(new_img, model_name)
                # color_moment_feature = ColorMoment.get_color_moment_features(new_img)
                # merged_color_moment = MergeFeatures.merge_color_moment(color_moment_feature)
                if final_features is None:
                    inner_img = new_img
                    final_features = np.array(features)
                else:
                    inner_img = inner_img + new_img
                    final_features = final_features + np.array(features)
                count = count + 1
        inner_img = inner_img/count
        Constants_p2.IMAGE_SET[id] = inner_img
        final_features = final_features / count
        dataset[id] = final_features
    # plot_images(dataset)
    return dataset


def construct_dataset_for_tasks(dataset, model_name, task_id):
    if task_id == "1" or task_id == "2":
        return extract_features_for_task_1_2(dataset, model_name)
        # return construct_dataset_for_task_1_2(dataset)
    else:
        return extract_features_for_task_3_4(dataset, model_name)
        # return construct_dataset_for_task_3_4(dataset)


def load_dataset_from_folder(path, task_id, model_name, subject_type_id, image_id, file_name, query_image_path, similarity_matrix_name,
                             type='.png'):
    # if image_id is None:
    #     load_dataset_from_folder_old(path, type)
    # else:
    if task_id == "1":
        return load_dataset_for_task1(path, model_name, subject_type_id, type)
    elif task_id == "2":
        return load_dataset_for_task2(path, model_name, subject_type_id, type)
    elif task_id == "3":
        return load_dataset_for_task3(path, model_name, type)
    elif task_id == "4":
        return load_dataset_for_task4(path, model_name, type)
    elif task_id == "5":
        return load_dataset_for_task5(path, file_name, image_id, query_image_path)
    elif task_id == "6":
        return load_dataset_for_task6(path, file_name, image_id, query_image_path)
    elif task_id == "7":
        return load_dataset_for_task7(path, file_name, image_id, query_image_path)
    elif task_id == "8" or task_id == "9":
        return load_dataset_for_task8_9(similarity_matrix_name)

def fetch_image(img_path):
    img = imread(img_path, as_gray=True)
    new_img = np.array(img).astype(np.float)
    GreyScaleNormalization.get_normalized_image(new_img)
    return img

def fetch_features_from_image_path(img_path, model_name):
    image_id = ntpath.splitext(ntpath.basename(img_path))[0]
    new_img = fetch_image(img_path)
    img = np.array(new_img).astype(np.float)
    GreyScaleNormalization.get_normalized_image(img)
    features = FeatureExtraction.get_features(img, model_name)
    return image_id, img, features


# Data is loaded from the target folder
def load_dataset_from_folder_old(path, model_name, image_id=None, query_image_path=None, type='.png'):
    dataset = {}
    # dataset['images'] = []
    # dataset['ids'] = []
    # col_dir = path + '/*' + type
    col_dir = path + '\\*' + type
    # print(col_dir)
    for img_path in glob.glob(col_dir):
        new_image_id, new_img, features = fetch_features_from_image_path(img_path, model_name)
        # image_id = ntpath.splitext(ntpath.basename(img_path))[0]
        # img = imread(img_path,as_gray=True)
        # new_img = np.array(img).astype(np.float)
        # GreyScaleNormalization.get_normalized_image(new_img)
        # features = FeatureExtraction.getFeatures(new_img, model_name)
        # color_moment_feature = ColorMoment.get_color_moment_features(new_img)
        # merged_color_moment = MergeFeatures.merge_color_moment(color_moment_feature)
        Constants_p2.IMAGE_SET[new_image_id] = new_img
        dataset[new_image_id] = features
        # dataset['ids'].append(image_id)
    # dataset['images'] = np.asarray(dataset['images'])
    # dataset['ids'] = np.asarray(dataset['ids'])
    if image_id is not None and image_id not in dataset:
        img_id, new_image_id, features = fetch_features_from_image_path(query_image_path, model_name)
        dataset[image_id] = features
    else:
        Constants_p2.QUERY_IMAGE_FOUND = True
    return dataset


def extract_subject_id_and_image_id(image_id, X):
    first_index = image_id.index(X) + len(X) + 1
    second_index = first_index + image_id[first_index:].index("-")
    third_index = second_index + 1
    return image_id[first_index: second_index], image_id[third_index:]


def extract_subject_id_image_type_and_second_id(image_id):
    first_index = image_id.index("-") + 1
    second_index = first_index + image_id[first_index:].index("-")
    third_index = second_index + 1 + image_id[second_index + 1:].index("-")
    fourth_index = third_index + 1
    return image_id[second_index + 1: third_index], image_id[first_index: second_index], image_id[fourth_index:]


def load_dataset_for_task1(path, model_name, X, type):
    dataset = {}
    col_dir = path + '\\*' + type
    for img_path in glob.glob(col_dir):
        if X not in img_path:
            continue
        image_id = ntpath.splitext(ntpath.basename(img_path))[0]
        subject_id, main_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
        if subject_id in dataset:
            dataset[subject_id].append(img_path)
        else:
            image_list = []
            image_list.append(img_path)
            dataset[subject_id] = image_list
    dataset = construct_dataset_for_tasks(dataset, model_name, "1")
    return dataset


def load_dataset_for_task2(path, model_name, Y, type):
    dataset = {}
    col_dir = path + '\\*' + type
    for img_path in glob.glob(col_dir):
        image_id = ntpath.splitext(ntpath.basename(img_path))[0]
        subject_id, main_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
        if subject_id != Y:
            continue
        if main_id in dataset:
            dataset[main_id].append(img_path)
        else:
            image_list = []
            image_list.append(img_path)
            dataset[main_id] = image_list
    dataset = construct_dataset_for_tasks(dataset, model_name, "2")
    return dataset


def load_dataset_for_task3(path, model_name, type, query_image_id=None, query_image_path=None):
    dataset = {}
    query_image_found = False
    col_dir = path + '\\*' + type
    for img_path in glob.glob(col_dir):
        image_id = ntpath.splitext(ntpath.basename(img_path))[0]
        if image_id == query_image_id:
            query_image_found = True
            Constants_p2.QUERY_IMAGE_FOUND = True
        subject_id, img_type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
        if img_type_id in dataset:
            if subject_id in dataset[img_type_id]:
                dataset[img_type_id][subject_id].append(img_path)
            else:
                image_list = []
                image_list.append(img_path)
                dataset[img_type_id][subject_id] = image_list
        else:
            image_list = []
            image_list.append(img_path)
            subject_dataset = {}
            subject_dataset[subject_id] = image_list
            dataset[img_type_id] = subject_dataset
    dataset = construct_dataset_for_tasks(dataset, model_name, "3")
    if query_image_id is not None:
        if query_image_found:
            query_image_path = path+"\\"+query_image_id+".png"
        new_image_id, new_img, features = fetch_features_from_image_path(query_image_path, model_name)
        dataset[query_image_id] = features
        Constants_p2.IMAGE_SET[query_image_id] = new_img
    return dataset


def load_dataset_for_task4(path, model_name, type, query_image_id=None, query_image_path=None):
    dataset = {}
    col_dir = path + '\\*' + type
    query_image_found = False
    for img_path in glob.glob(col_dir):
        image_id = ntpath.splitext(ntpath.basename(img_path))[0]
        if image_id == query_image_id:
            query_image_found = True
            Constants_p2.QUERY_IMAGE_FOUND = True
        subject_id, img_type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)

        if subject_id in dataset:
            if img_type_id in dataset[subject_id]:
                dataset[subject_id][img_type_id].append(img_path)
            else:
                image_list = []
                image_list.append(img_path)
                dataset[subject_id][img_type_id] = image_list
        else:
            image_list = []
            image_list.append(img_path)
            type_dataset = {}
            type_dataset[img_type_id] = image_list
            dataset[subject_id] = type_dataset
    dataset = construct_dataset_for_tasks(dataset, model_name, "4")
    if query_image_id is not None:
        if query_image_found:
            query_image_path = path+"\\"+query_image_id+".png"
        new_image_id, new_img, features = fetch_features_from_image_path(query_image_path, model_name)
        dataset[query_image_id] = features
        Constants_p2.IMAGE_SET[query_image_id] = new_img
    return dataset


def extract_task_id_model_name_reduction_name(file_name):
    first_index = file_name.index("_") + 8
    second_index = first_index + file_name[first_index:].index("_")
    third_index = second_index + 1
    fourth_index = third_index + file_name[third_index:].index("_")
    fifth_index = fourth_index + 1
    sixth_index = fifth_index + file_name[fifth_index:].index(".")
    return file_name[first_index: second_index], file_name[third_index: fourth_index], file_name[
                                                                                       fifth_index:sixth_index]


def load_dataset_for_task5(images_path, file_name, image_id, query_image_path):
    task_id, model_name, reduction_name = extract_task_id_model_name_reduction_name(file_name)
    transformation_matrix = ContentReader.get_transformation_matrix(file_name)
    features_dataset = load_dataset_from_folder_old(images_path, model_name, image_id, query_image_path)
    dataset = Transformation.get_latent_features(features_dataset, transformation_matrix)
    return dataset


def load_dataset_for_task6(images_path, file_name, image_id, query_image_path):
    task_id, model_name, reduction_name = extract_task_id_model_name_reduction_name(file_name)
    transformation_matrix = ContentReader.get_transformation_matrix(file_name)
    type = '.png'
    features_dataset = load_dataset_for_task3(images_path, model_name, type, image_id, query_image_path)
    dataset = Transformation.get_latent_features(features_dataset, transformation_matrix)
    return dataset


def load_dataset_for_task7(images_path, file_name, image_id, query_image_path):
    task_id, model_name, reduction_name = extract_task_id_model_name_reduction_name(file_name)
    transformation_matrix = ContentReader.get_transformation_matrix(file_name)
    type = '.png'
    features_dataset = load_dataset_for_task4(images_path, model_name, type, image_id, query_image_path)
    dataset = Transformation.get_latent_features(features_dataset, transformation_matrix)
    return dataset

def load_dataset_for_task8_9(similarity_matrix_name):
    matrix = ContentReader.get_matrix(similarity_matrix_name)
    return matrix
