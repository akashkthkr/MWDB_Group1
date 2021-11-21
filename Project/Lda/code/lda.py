import pickle
from time import time
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from Project.Lda.code import dataset, features, constants_lda, dummy_data, lda_inference
import json
from skimage.io import imread_collection, imread

from collections import Counter

type_id_mapping = ['cc', 'con', 'emboss', 'jitter', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth',
                   'stipple']


def get_type_from_filename(filename):
    return filename.split('-')[1]


def get_subject_from_filename(filename):
    return filename.split('-')[2]


def normalize_rows(mat):
    norm_mat = mat - np.min(mat, axis=1)[:, np.newaxis]
    norm_mat = norm_mat / norm_mat.sum(axis=1)[:, np.newaxis]
    return norm_mat


def get_trained_lda(n_components, object_feature_mapping):
    print("LDA for n_components=", n_components)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=100,
                                    learning_method='online',
                                    learning_offset=10.,
                                    random_state=7,
                                    n_jobs=-1,
                                    evaluate_every=5,
                                    perp_tol=0.01,
                                    verbose=1)
    lda.fit(object_feature_mapping)

    print("components=", lda.components_)
    print("components shape=", lda.components_.shape)

    return lda


def get_type_feature_mapping(json_data, image_ids, feature_type, num_types, num_features):
    type_feature_mapping = np.zeros((num_types, num_features))

    prob_type = np.zeros(num_types)

    for id in image_ids:
        type_id = type_id_mapping.index(id.split('-')[1])
        type_feature_mapping[type_id] += json_data[id]['features'][feature_type]
        prob_type[type_id] += 1

    print("type_sum=", prob_type)
    prob_type /= np.sum(prob_type)
    print("prob_type=", prob_type, "sum=", np.sum(prob_type), " len=", len(prob_type))
    return type_feature_mapping, prob_type


def get_subject_feature_mapping(json_data, image_ids, feature_type, num_subjects, num_features):
    subject_feature_mapping = np.zeros((num_subjects, num_features))

    prob_subject = np.zeros(num_subjects)

    for id in image_ids:
        subject_id = int(id.split('-')[2]) - 1
        subject_feature_mapping[subject_id] += json_data[id]['features'][feature_type]
        prob_subject[subject_id] += 1

    print("subject_sum=", prob_subject)
    prob_subject /= np.sum(prob_subject)
    print("prob_subject=", prob_subject, "sum=", np.sum(prob_subject), " len=", len(prob_subject))
    return subject_feature_mapping, prob_subject


def get_prob_feature_given_type(type_feature_mapping, num_types, num_features):
    prob_feature_given_type = np.zeros((num_types, num_features))
    # prob_feature_given_type = np.zeros((len(type_id_mapping), object_feature_mapping.shape[1]))
    for type_id, type_arr in enumerate(type_feature_mapping):
        for feature_id, feature_arr in enumerate(type_arr):
            prob_feature_given_type[type_id][feature_id] = type_feature_mapping[type_id][feature_id] / np.sum(
                type_feature_mapping[type_id])
        print("prob_feature_given_type_", type_id, "=", np.sum(prob_feature_given_type[type_id]))
    print("type_feature_probs=", prob_feature_given_type)
    return prob_feature_given_type


def get_prob_feature_given_subject(subject_feature_mapping, num_subjects, num_features):
    prob_feature_given_subject = np.zeros((num_subjects, num_features))
    # prob_feature_given_subject = np.zeros((len(type_id_mapping), object_feature_mapping.shape[1]))
    for sub_id, sub_arr in enumerate(subject_feature_mapping):
        for feature_id, feature_arr in enumerate(sub_arr):
            prob_feature_given_subject[sub_id][feature_id] = subject_feature_mapping[sub_id][feature_id] / np.sum(
                subject_feature_mapping[sub_id])
    print("subject_feature_probs=", prob_feature_given_subject)
    return prob_feature_given_subject


def get_prob_type_given_feature_in_topic(lda, prob_type, prob_feature_given_type, n_components, num_features,
                                         num_types):
    prob_feature_in_topic = np.array(
        [np.sum(lda.components_[:, feature_id]) for feature_id, _ in enumerate(lda.components_[0])])
    prob_feature_in_topic /= np.sum(prob_feature_in_topic)
    print("feature probabilities in component=", prob_feature_in_topic, "sum=", np.sum(prob_feature_in_topic))
    prob_type_given_feature_in_topic = np.zeros((n_components, num_features, num_types))
    # prob_type_given_feature_in_topic = np.zeros((num_com, object_feature_mapping.shape[1],len(type_feature_mapping)))
    for component_id, component in enumerate(lda.components_):
        for feature_id, feature in enumerate(component):
            prob_type_given_feature_in_topic[component_id][feature_id] = [
                prob_feature_given_type[type_id][feature_id] * prob_type[type_id] / prob_feature_in_topic[feature_id]
                for type_id, type in enumerate(type_id_mapping)]
    return prob_type_given_feature_in_topic


def get_prob_subject_given_feature_in_topic(lda, prob_subject, prob_feature_given_subject, n_components, num_features,
                                            num_subjects):
    prob_feature_in_topic = np.array(
        [np.sum(lda.components_[:, feature_id]) for feature_id, _ in enumerate(lda.components_[0])])
    prob_feature_in_topic /= np.sum(prob_feature_in_topic)
    print("feature probabilities in component=", prob_feature_in_topic, "sum=", np.sum(prob_feature_in_topic))
    prob_subject_given_feature_in_topic = np.zeros((n_components, num_features, num_subjects))
    # prob_type_given_feature_in_topic = np.zeros((num_com, object_feature_mapping.shape[1],len(type_feature_mapping)))
    for component_id, component in enumerate(lda.components_):
        for feature_id, feature in enumerate(component):
            prob_subject_given_feature_in_topic[component_id][feature_id] = [
                prob_feature_given_subject[subject_id][feature_id] * prob_subject[subject_id] / prob_feature_in_topic[
                    feature_id]
                for subject_id in range(num_subjects)]
    return prob_subject_given_feature_in_topic


def get_components_by_type(lda, prob_type_given_feature_in_topic, n_components, num_types):
    prob_lda_topics = np.sum(lda.components_, axis=1) / np.sum(lda.components_)
    print("topic probabilities=", prob_lda_topics, "sum=", np.sum(prob_lda_topics))
    prob_lda_components_ = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    print("components_probs=", prob_lda_components_, " sum=", np.sum(prob_lda_components_, axis=1), )

    component_by_type = np.zeros((n_components, num_types))
    for component_id, component in enumerate(lda.components_):
        for type_id, type in enumerate(type_id_mapping):
            component_by_type[component_id][type_id] = prob_lda_topics[component_id] * np.sum(
                prob_lda_components_[component_id] * prob_type_given_feature_in_topic[component_id][:, type_id])
    component_by_type = np.nan_to_num(component_by_type)
    print("component_by_type=", component_by_type, "shape=", component_by_type.shape, "topic scores=",
          np.sum(component_by_type, axis=1), np.sum(component_by_type))
    return component_by_type


def get_components_by_subject(lda, prob_subject_given_feature_in_topic, n_components, num_subjects):
    prob_lda_topics = np.sum(lda.components_, axis=1) / np.sum(lda.components_)
    print("topic probabilities=", prob_lda_topics, "sum=", np.sum(prob_lda_topics))
    prob_lda_components_ = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    print("components_probs=", prob_lda_components_, " sum=", np.sum(prob_lda_components_, axis=1))

    component_by_subject = np.zeros((n_components, num_subjects))
    for component_id, component in enumerate(lda.components_):
        for subject_id in range(num_subjects):
            component_by_subject[component_id][subject_id] = prob_lda_topics[component_id] * np.sum(
                prob_lda_components_[component_id] * prob_subject_given_feature_in_topic[component_id][:, subject_id])
    component_by_subject = np.nan_to_num(component_by_subject)
    print("component_by_subject=", component_by_subject, "shape=", component_by_subject.shape, "topic scores=",
          np.sum(component_by_subject, axis=1), np.sum(component_by_subject))
    return component_by_subject


def rank_components(component, mapping):
    ranked_components = [topic.argsort()[::-1] for topic_idx, topic in enumerate(component)]
    if mapping == 'type':
        ranked_components = [
            ("topic_" + str(topic_id), [(type_id_mapping[id], component[topic_id][id]) for id in topic]) for
            topic_id, topic in enumerate(ranked_components)]
    elif mapping == 'subject':
        ranked_components = [
            ("topic_" + str(topic_id), [('subject_' + str(id), component[topic_id][id]) for id in topic]) for
            topic_id, topic in enumerate(ranked_components)]
    else:
        print('invalid mapping ' + mapping)
        raise
    print("ranked_components", ranked_components)
    return ranked_components


def save_lda(filename, lda):
    with open(filename, 'wb') as pickle_file:
        # with open('lda_' + feature_type + '_' + str(top_features_count) + '.pk', 'wb') as pickle_file:
        pickle.dump(lda, pickle_file)
    # pickle.dumps(lda,)


def save_to_json(filename, object_to_store):
    with open(filename, 'w') as json_file:
        # with open('latent_semantics_' + feature_type + '_' + str(top_features_count) + '.json', 'w') as json_file:
        json.dump(object_to_store, json_file)


def save_to_pickle(filename, object_to_store):
    with open(filename, 'wb') as pickle_file:
        # with open('lda_transformed_dataset_' + feature_type + '_' + str(top_features_count) + '.pk', 'wb') as pickle_file:
        # pickle.dump(lda.transform(object_feature_mapping),pickle_file)
        pickle.dump(object_to_store, pickle_file)


def get_num_subjects(image_ids):
    subs = set()
    for id in image_ids:
        sub_id = int(id.split('-')[2])
        subs.add(sub_id)
    return max(subs) + 1


# Transformation matrix creation
def get_prob_feature_in_topics(lda_filename, num_features, prob_feature_given_type):
    with open(lda_filename, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)
        components = lda.components_
        num_topics = components.shape[0]
        prob_lda_topics = np.sum(lda.components_, axis=1) / np.sum(lda.components_)
        print("topic probabilities=", prob_lda_topics, "sum=", np.sum(prob_lda_topics))
        prob_lda_components_ = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        print("components_probs=", prob_lda_components_, " sum=", np.sum(prob_lda_components_, axis=1))
        prob_feature_in_topics = np.zeros((num_topics, num_features))
        for topic_id, topic in enumerate(prob_lda_components_):
            for feature_id in range(num_features):
                # prob_feature_in_topics[topic_id][feature_id] = prob_lda_topics[topic_id]*np.dot(
                # prob_lda_components_[topic_id], prob_feature_given_type[:,feature_id])
                prob_feature_in_topics[topic_id][feature_id] = np.dot(prob_lda_components_[topic_id],
                                                                      prob_feature_given_type[:, feature_id])
        print('prob_feature_in_topics', prob_feature_in_topics, np.sum(prob_feature_in_topics), "shape=",
              prob_feature_in_topics.shape)
    return prob_feature_in_topics


def get_top_n(latent_semantics_filename, query_image_name, n):
    query_image = imread(query_image_name) / 255
    feature_dict = features.get_all_features(query_image)

    if 'task1' in latent_semantics_filename:
        transformed_data = []
        trans_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task1.pk'
        with open(trans_data_filename, 'rb') as pickle_file:
            transformed_data = pickle.load(pickle_file)

        image_ids = []
        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task1.pk'
        with open(imageids_filename, 'rb') as pickle_file:
            image_ids = pickle.load(pickle_file)

        type = None
        meta_filename = constants_lda.OUTPUT + 'lda_meta_task1.pk'
        with open(meta_filename, 'rb') as pickle_file:
            type = pickle.load(pickle_file)

        query_feature = feature_dict[type]
        lda_filename = constants_lda.OUTPUT + 'lda_task1.pk'
        transformed_vector = lda_inference.infer(lda_filename, [query_feature])[0]
        topn = lda_inference.get_n_closest_images(transformed_vector, transformed_data, n, image_ids)
        return topn

    elif 'task2' in latent_semantics_filename:
        transformed_data = []
        trans_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task2.pk'
        with open(trans_data_filename, 'rb') as pickle_file:
            transformed_data = pickle.load(pickle_file)

        image_ids = []
        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task2.pk'
        with open(imageids_filename, 'rb') as pickle_file:
            image_ids = pickle.load(pickle_file)

        type = None
        meta_filename = constants_lda.OUTPUT + 'lda_meta_task2.pk'
        with open(meta_filename, 'rb') as pickle_file:
            type = pickle.load(pickle_file)

        query_feature = feature_dict[type]
        lda_filename = constants_lda.OUTPUT + 'lda_task2.pk'
        transformed_vector = lda_inference.infer(lda_filename, [query_feature])[0]
        topn = lda_inference.get_n_closest_images(transformed_vector, transformed_data, n, image_ids)
        return topn
    elif 'task3' in latent_semantics_filename:
        transformed_data = []
        trans_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task3.pk'
        with open(trans_data_filename, 'rb') as pickle_file:
            transformed_data = pickle.load(pickle_file)

        image_ids = []
        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task3.pk'
        with open(imageids_filename, 'rb') as pickle_file:
            image_ids = pickle.load(pickle_file)

        type = None
        meta_filename = constants_lda.OUTPUT + 'lda_meta_task3.pk'
        with open(meta_filename, 'rb') as pickle_file:
            type = pickle.load(pickle_file)

        trans_matrix = []
        transformation_matrix_filename = constants_lda.OUTPUT + 'lda_transformation_matrix_task3.pk'
        with open(transformation_matrix_filename, 'rb') as pickle_file:
            trans_matrix = pickle.load(pickle_file)

        query_feature = feature_dict[type]
        transformed_vector = np.dot([query_feature], trans_matrix)[0]
        transformed_vector = normalize_rows(np.array([transformed_vector]))[0]

        topn = lda_inference.get_n_closest_images(transformed_vector, transformed_data, n, image_ids)
        return topn

    elif 'task4' in latent_semantics_filename:
        transformed_data = []
        trans_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task4.pk'
        with open(trans_data_filename, 'rb') as pickle_file:
            transformed_data = pickle.load(pickle_file)

        image_ids = []
        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task4.pk'
        with open(imageids_filename, 'rb') as pickle_file:
            image_ids = pickle.load(pickle_file)

        type = None
        meta_filename = constants_lda.OUTPUT + 'lda_meta_task4.pk'
        with open(meta_filename, 'rb') as pickle_file:
            type = pickle.load(pickle_file)

        trans_matrix = []
        transformation_matrix_filename = constants_lda.OUTPUT + 'lda_transformation_matrix_task4.pk'
        with open(transformation_matrix_filename, 'rb') as pickle_file:
            trans_matrix = pickle.load(pickle_file)

        query_feature = feature_dict[type]
        transformed_vector = np.dot([query_feature], trans_matrix)[0]
        transformed_vector = normalize_rows(np.array([transformed_vector]))[0]

        topn = lda_inference.get_n_closest_images(transformed_vector, transformed_data, n, image_ids)
        return topn
    else:
        print("Invalid latent semantics filename")


class LdaClass:
    def __init__(self, feature_type, dataset_json_filename):
        if feature_type is None:
            self.feature_type = "combined"
        else:
            self.feature_type = feature_type
        self.json_data = dataset.open_json(dataset_json_filename)
        self.image_ids = self.json_data.keys()
        self.data_matrix = np.array([self.json_data[id]['features'][self.feature_type] for id in self.image_ids])
        self.num_types = len(type_id_mapping)
        self.num_subjects = get_num_subjects(self.image_ids)
        self.num_features = self.data_matrix.shape[1]

    def task1(self, feature_type, n_components, X):
        json_filename = dataset.process_data('task1', X)
        json_data = dataset.open_json(json_filename)
        image_ids = json_data.keys()
        num_subjects = get_num_subjects(image_ids)
        print(len(image_ids))
        print("num_subjects=", num_subjects)
        object_feature_mapping = np.array([json_data[id]['features'][feature_type] for id in image_ids])
        lda = get_trained_lda(n_components, object_feature_mapping)
        num_features = object_feature_mapping.shape[1]
        print("object_feature_mapping=", object_feature_mapping, ' shape=', object_feature_mapping.shape)
        subject_feature_mapping, prob_subject = get_subject_feature_mapping(json_data, image_ids, feature_type,
                                                                            num_subjects, num_features)
        prob_feature_given_subject = get_prob_feature_given_subject(subject_feature_mapping, num_subjects, num_features)
        prob_subject_given_feature_in_topic = get_prob_subject_given_feature_in_topic(lda, prob_subject,
                                                                                      prob_feature_given_subject,
                                                                                      n_components, num_features,
                                                                                      num_subjects)
        components_by_subject = get_components_by_subject(lda, prob_subject_given_feature_in_topic, n_components,
                                                          num_subjects)
        ranked_components_by_subject = rank_components(components_by_subject, 'subject')

        lda_filename = constants_lda.OUTPUT + 'lda_task1.pk'
        save_lda(lda_filename, lda)
        print("Saving LDA model to", lda_filename)

        latent_semantics_filename = constants_lda.OUTPUT + 'lda_latent_semantics_task1.json'
        save_to_json(latent_semantics_filename, ranked_components_by_subject)
        print("Saving latent semantics to file", latent_semantics_filename)

        transformed = lda_inference.infer(lda_filename, self.data_matrix)
        print("linear transformed=", transformed, " sum row=", np.sum(transformed))

        transformed_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task1.pk'
        save_to_pickle(transformed_data_filename, transformed)

        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task1.pk'
        save_to_pickle(imageids_filename, list(self.image_ids))

        print("Saving trsnsformed dataset to file ", transformed_data_filename)

        metadata_filename = constants_lda.OUTPUT + 'lda_meta_task1.pk'
        save_to_pickle(metadata_filename, feature_type)

        return lda_filename, latent_semantics_filename

    def task2(self, feature_type, n_components, Y):
        json_filename = dataset.process_data('task2', Y)
        num_types = len(type_id_mapping)
        json_data = dataset.open_json(json_filename)
        image_ids = json_data.keys()
        print(len(image_ids))
        object_feature_mapping = np.array([json_data[id]['features'][feature_type] for id in image_ids])
        lda = get_trained_lda(n_components, object_feature_mapping)
        num_features = object_feature_mapping.shape[1]
        print("object_feature_mapping=", object_feature_mapping, ' shape=', object_feature_mapping.shape)
        type_feature_mapping, prob_type = get_type_feature_mapping(json_data, image_ids, feature_type, num_types,
                                                                   num_features)
        prob_feature_given_type = get_prob_feature_given_type(type_feature_mapping, num_types, num_features)
        prob_type_given_feature_in_topic = get_prob_type_given_feature_in_topic(lda, prob_type, prob_feature_given_type,
                                                                                n_components, num_features, num_types)
        components_by_type = get_components_by_type(lda, prob_type_given_feature_in_topic, n_components, num_types)
        ranked_components_by_type = rank_components(components_by_type, 'type')

        lda_filename = constants_lda.OUTPUT + 'lda_task2.pk'
        save_lda(lda_filename, lda)
        print("Saving LDA model to file ", lda_filename)

        latent_semantics_filename = constants_lda.OUTPUT + 'lda_latent_semantics_task2.json'
        save_to_json(latent_semantics_filename, ranked_components_by_type)
        print("Saving latent semantics to file", latent_semantics_filename)

        transformed = lda_inference.infer(lda_filename, self.data_matrix)
        print("linear transformed=", transformed, " sum row=", np.sum(transformed))

        transformed_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task2.pk'
        save_to_pickle(transformed_data_filename, transformed)
        print("Saving trsnsformed dataset to file ", transformed_data_filename)

        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task2.pk'
        save_to_pickle(imageids_filename, list(self.image_ids))

        metadata_filename = constants_lda.OUTPUT + 'lda_meta_task2.pk'
        save_to_pickle(metadata_filename, feature_type)

        return lda_filename, latent_semantics_filename, transformed_data_filename

    def task3(self, type_type_similarity_matrix, n_components):
        num_types = len(type_id_mapping)
        num_features_types = type_type_similarity_matrix.shape[1]
        lda = get_trained_lda(n_components, type_type_similarity_matrix)

        # prob_feature_given_type = get_prob_feature_given_type(type_type_similarity_matrix,num_types,num_features_types)
        ranked_components_by_type = rank_components(lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis],
                                                    'type')

        lda_filename = constants_lda.OUTPUT + 'lda_task3.pk'
        save_lda(lda_filename, lda)
        print("Saving LDA model to", lda_filename)

        latent_semantics_filename = constants_lda.OUTPUT + 'lda_latent_semantics_task3.json'
        save_to_json(latent_semantics_filename, ranked_components_by_type)
        print("Saving latent semantics to file ", latent_semantics_filename)

        type_feature_mapping, _ = get_type_feature_mapping(self.json_data, self.image_ids, self.feature_type, num_types,
                                                           self.num_features)
        trans_matrix = lda_inference.transformation_matrix(lda_filename, type_feature_mapping)
        print("transformation_matrix=", trans_matrix)
        trans_matrix_filename = constants_lda.OUTPUT + 'lda_transformation_matrix_task3.pk'
        save_to_pickle(trans_matrix_filename, trans_matrix)
        print('Saved transformation matrix to :' + trans_matrix_filename)

        transformed = np.dot(self.data_matrix, trans_matrix)
        transformed = normalize_rows(transformed)
        # print("transformed=",transformed," sum row=",np.sum(transformed[0]))
        print("linear transformed=", transformed, " sum row=", np.sum(transformed))
        transformed_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task3.pk'
        save_to_pickle(transformed_data_filename, transformed)
        print("Saving trsnsformed dataset to file ", transformed_data_filename)

        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task3.pk'
        save_to_pickle(imageids_filename, list(self.image_ids))

        metadata_filename = constants_lda.OUTPUT + 'lda_meta_task3.pk'
        save_to_pickle(metadata_filename, self.feature_type)

        return lda_filename, latent_semantics_filename, transformed_data_filename, trans_matrix_filename

    def task4(self, subject_subject_similarity_matrix, n_components):

        num_subjects = subject_subject_similarity_matrix.shape[0]
        num_features_subjects = subject_subject_similarity_matrix.shape[1]
        lda = get_trained_lda(n_components, subject_subject_similarity_matrix)

        # prob_feature_given_subject = get_prob_feature_given_subject(subject_subject_similarity_matrix,num_subjects,num_features_subjects)
        ranked_components_by_subject = rank_components(lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis],
                                                       'subject')

        lda_filename = constants_lda.OUTPUT + 'lda_task4.pk'
        save_lda(lda_filename, lda)
        print("Saving LDA model to", lda_filename)

        latent_semantics_filename = constants_lda.OUTPUT + 'lda_latent_semantics_task4.json'
        save_to_json(latent_semantics_filename, ranked_components_by_subject)
        print("Saving latent semantics to file ", latent_semantics_filename)

        subject_feature_mapping, _ = get_subject_feature_mapping(self.json_data, self.image_ids, self.feature_type,
                                                                 num_subjects, self.num_features)
        trans_matrix = lda_inference.transformation_matrix(lda_filename, subject_feature_mapping)
        print("transformation_matrix=", trans_matrix)
        trans_matrix_filename = constants_lda.OUTPUT + 'lda_transformation_matrix_task4.pk'
        save_to_pickle(trans_matrix_filename, trans_matrix)
        print('Saved transformation matrix to :' + trans_matrix_filename)

        transformed = np.dot(self.data_matrix, trans_matrix)
        transformed = normalize_rows(transformed)
        # print("transformed=",transformed," sum row=",np.sum(transformed[0]))
        print("linear transformed=", transformed, " sum row=", np.sum(transformed))
        transformed_data_filename = constants_lda.OUTPUT + 'lda_transformed_dataset_task4.pk'
        save_to_pickle(transformed_data_filename, transformed)
        print("Saving trsnsformed dataset to file ", transformed_data_filename)

        imageids_filename = constants_lda.OUTPUT + 'lda_imageids_task4.pk'
        save_to_pickle(imageids_filename, list(self.image_ids))

        metadata_filename = constants_lda.OUTPUT + 'lda_meta_task4.pk'
        save_to_pickle(metadata_filename, self.feature_type)

        return lda_filename, latent_semantics_filename, transformed_data_filename, trans_matrix_filename

    def task5(self, latent_semantics_filename, query_image_name, n):
        topn = get_top_n(latent_semantics_filename, query_image_name, n)
        return topn

    def task6(self, latent_semantics_filename, query_image_name):
        n = 10
        topn = get_top_n(latent_semantics_filename, query_image_name, n)
        types = [get_type_from_filename(filename) for filename in topn]
        counter = Counter(types)
        top_types = counter.most_common()
        print('top_types = ', top_types)
        return top_types

    def task7(self, latent_semantics_filename, query_image_name):
        n = 30
        topn = get_top_n(latent_semantics_filename, query_image_name, n)
        types = [get_subject_from_filename(filename) for filename in topn]
        counter = Counter(types)
        top_types = counter.most_common()
        print('top_subjects = ', top_types)
        return top_types


def main():
    lda_ob = LdaClass('cm', 'all_all_database.json')
    lda_ob.task1('cm', 10, 'con')
    lda_ob.task2('cm', 5, '3')

    lda_ob = LdaClass('combined', 'all_all_database.json')
    lda_ob.task3(np.array(dummy_data.type_type_similarity_matrix), 30)
    lda_ob.task4(np.array(np.array(dummy_data.subject_subject_similarity_matrix)), 20)

    topn = lda_ob.task5('../Outputs/lda_latent_semantics_task1', '../inputs/query_images/image-cc-1-2.png', 5)
    topn = lda_ob.task5('../Outputs/lda_latent_semantics_task2', '../inputs/query_images/image-cc-1-2.png', 5)
    topn = lda_ob.task5('../Outputs/lda_latent_semantics_task3', '../inputs/query_images/image-cc-1-2.png', 5)
    topn = lda_ob.task5('../Outputs/lda_latent_semantics_task4', '../inputs/query_images/image-cc-1-2.png', 5)

    topn = lda_ob.task6('../Outputs/lda_latent_semantics_task1', '../inputs/query_images/image-cc-1-2.png')
    topn = lda_ob.task6('../Outputs/lda_latent_semantics_task2', '../inputs/query_images/image-cc-1-2.png')
    topn = lda_ob.task6('../Outputs/lda_latent_semantics_task3', '../inputs/query_images/image-cc-1-2.png')
    topn = lda_ob.task6('../Outputs/lda_latent_semantics_task4', '../inputs/query_images/image-cc-1-2.png')

    topn = lda_ob.task7('../Outputs/lda_latent_semantics_task1', '../inputs/query_images/image-cc-1-2.png')
    topn = lda_ob.task7('../Outputs/lda_latent_semantics_task2', '../inputs/query_images/image-cc-1-2.png')
    topn = lda_ob.task7('../Outputs/lda_latent_semantics_task3', '../inputs/query_images/image-cc-1-2.png')
    topn = lda_ob.task7('../Outputs/lda_latent_semantics_task4', '../inputs/query_images/image-cc-1-2.png')


if __name__ == "__main__":
    main()

# top_features_ind = [sorted(topic.argsort()[:-top_features_count-1:-1]) for topic_idx, topic in enumerate(lda.components_)]
# print("top_features_ind=",top_features_ind)
# print("inference objects shape=",np.array(object_feature_mapping[0:110:3]).shape)
# transform = lda.transform(object_feature_mapping[0:110:3])
# print("params=",lda.get_params())
# print("transform=",transform)
# print("transform shape=",np.array(transform).shape)
# print("done in %0.3fs." % (time() - t0))
