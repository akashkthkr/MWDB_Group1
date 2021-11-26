import numpy
import json
from numpyencoder import NumpyEncoder
from Phase3.decision_tree.decision_tree3 import DecisionTree
from Project.services.ImageFetchService import extract_subject_id_image_type_and_second_id
from constants.Constants_Phase3 import OUTPUTS_PATH
from sklearn import tree
from Phase3.SVM.SVMUtilities import SVMUtilities



TASK_LABEL_MAPPING = {
    1: "X labels",
    2: "Y labels",
    3: "Z labels"
}


class DecisionTreeHandler:
    def __init__(self, task_id=None, train_data=None, test_data=None):
        self.task_id = task_id
        self.train_data = train_data
        self.test_data = test_data
        self.labels = self.get_labels()
        self.false_positives, self.false_negatives, self.true_positives, self.true_negatives = {}, {}, {}, {}

    def execute(self):
        images_assosciation = {}
        negative_image_assosciation = {}
        clf = DecisionTree()
        count = 1
        for original_type_id in self.get_labels():
            print(original_type_id)
            print(count)
            count = count + 1
            training_image_ids, training_set_X, training_set_Y = [], [], []
            for image_id in self.train_data:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                training_image_ids.append(image_id)
                training_set_X.append(self.train_data[image_id])
                if type_id == original_type_id:
                    training_set_Y.append(1)
                else:
                    training_set_Y.append(0)

            test_image_ids, test_set_X = [], []

            for image_id in self.test_data:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                test_image_ids.append(image_id)
                test_set_X.append(self.test_data[image_id])
            clf.fit(training_set_X, training_set_Y)
            print("len::: {}   training settt XXX : {}".format(len(training_set_X), training_set_X))
            print("len::: {}   training settt YYY: {}".format(len(training_set_Y), training_set_Y))
            print("treeeee : {}".format(clf.tree))
            prediction_values = clf.predict(numpy.array(test_set_X))
            print("prediction_values:::: {}".format(prediction_values))
            predictions = numpy.sign(prediction_values)

            index = 0
            for image_id in test_image_ids:
                if predictions[index] == 1:
                    if original_type_id in image_id:
                        if original_type_id in self.true_positives:
                            self.true_positives[original_type_id].append(image_id)
                        else:
                            true_positive_list = [image_id]
                            if len(self.true_positives) == 0:
                                self.true_positives = {original_type_id: true_positive_list}
                            else:
                                self.true_positives[original_type_id] = true_positive_list
                    else:
                        if original_type_id in self.false_positives:
                            self.false_positives[original_type_id].append(image_id)
                        else:
                            false_positive_list = [image_id]
                            if len(self.false_positives) == 0:
                                self.false_positives = {original_type_id: false_positive_list}
                            else:
                                self.false_positives[original_type_id] = false_positive_list

                    if image_id in images_assosciation:
                        if any(original_type_id in type for type in images_assosciation[image_id]):
                            for value in images_assosciation[image_id]:
                                value[original_type_id] = max(value[original_type_id], prediction_values[index])
                        else:
                            y = {original_type_id: prediction_values[index]}
                            images_assosciation[image_id].append(y)
                    else:
                        x = {original_type_id: prediction_values[index]}
                        images_assosciation[image_id] = [x]
                else:
                    if original_type_id in image_id:
                        if original_type_id in self.false_negatives:
                            self.false_negatives[original_type_id].append(image_id)
                        else:
                            false_negative_list = [image_id]
                            if len(self.false_negatives) == 0:
                                self.false_negatives = {original_type_id: false_negative_list}
                            else:
                                self.false_negatives[original_type_id] = false_negative_list
                    else:
                        if original_type_id in self.true_negatives:
                            self.true_negatives[original_type_id].append(image_id)
                        else:
                            true_negative_list = [image_id]
                            if len(self.true_negatives) == 0:
                                self.true_negatives = {original_type_id: true_negative_list}
                            else:
                                self.true_negatives[original_type_id] = true_negative_list

                    if image_id in negative_image_assosciation:
                        if any(original_type_id in type for type in negative_image_assosciation[image_id]):
                            for value in negative_image_assosciation[image_id]:
                                value[original_type_id] = max(value[original_type_id], prediction_values[index])
                        else:
                            y = {original_type_id: prediction_values[index]}
                            negative_image_assosciation[image_id].append(y)
                    else:
                        x = {original_type_id: prediction_values[index]}
                        negative_image_assosciation[image_id] = [x]
                index = index + 1

        classifier_results = {}

        for image_id in images_assosciation:
            ans = -1
            for X in images_assosciation[image_id]:
                for key in X:
                    if X[key] > ans:
                        classifier_results[image_id] = key
                        ans = X[key]
        remaining_images = []

        for image_id in self.test_data:
            if image_id not in classifier_results:
                remaining_images.append(image_id)
                ans = -1000000000
                for X in negative_image_assosciation[image_id]:
                    for key in X:
                        if X[key] > ans:
                            classifier_results[image_id] = key
                            ans = X[key]

        jsonString = json.dumps(images_assosciation, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + "DT_Classifier_Values_Task1.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        jsonString = json.dumps(negative_image_assosciation, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + "DT_Classifier_negative_Values_Task1.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        jsonString = json.dumps(classifier_results, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + "DT_Classifier_Results_Task1.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        svm_utilities = SVMUtilities(self.false_positives, self.true_positives, self.false_negatives,
                                     self.true_negatives, self.get_labels(), self.task_id)
        svm_utilities.save_fp_misses_rate()
        print(remaining_images)
        pass

    def execute1(self):
        decision_tree_classifier = DecisionTree()
        # for image in self.train_data:
        #     print(image)
        train_data = list()
        labels_data = list()
        for image, features in self.train_data.items():
            train_data.append(list(features))
            labels_data.append(self.get_label_from_image(image))

        # print("debugger")
        decision_tree_classifier.fit(train_data, labels_data)

        test_data = list()
        for image, features in self.test_data.items():
            test_data.append(features)

        print("library")
        resultss = list()
        clf = tree.DecisionTreeClassifier(max_depth=len(train_data))
        clf = clf.fit(numpy.array(train_data), numpy.array(labels_data))
        resultss = clf.predict(numpy.array(test_data))
        false_positivess, misses = self.calculate_misses_and_false_positives(resultss)
        print("resultsss::: {}".format(resultss))
        print("False Positivessss:::: {}".format(false_positivess))
        print("lennn   {}:  ,   )False Positives:::: {}".format(len(false_positivess), false_positivess))

        # print("for test data")
        results = decision_tree_classifier.predict(test_data)
        print("results::: {}".format(results))
        false_positives, misses = self.calculate_misses_and_false_positives(results)
        print("len   {}:  ,   )False Positives:::: {}".format(len(false_positives), false_positives))

        # print("\n\ntrain data")
        # results1 = decision_tree_classifier.predict(train_data)
        # false_positives1, misses = self.calculate_misses_and_false_positives(results1)
        # print("False Positives111:::: {}".format(false_positives1))

        # print("library")
        # # clf = tree.DecisionTreeClassifier()
        # # clf = clf.fit(numpy.array(train_data), numpy.array(labels_data))
        # resultss1 = clf.predict(numpy.array(train_data))
        # false_positivess1, misses = self.calculate_misses_and_false_positives(resultss1)
        # print("False Positivessss:::: {}".format(false_positivess1))

    def get_labels(self):
        if self.task_id == "1":
            return {"cc", "con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot",
                    "smooth", "stipple"}
        elif self.task_id == "2":
            return set([i for i in range(1, 41, 1)])
        elif self.task_id == "3":
            return set([i for i in range(1, 11, 1)])
        else:
            raise(Exception("Decision Tree will not be supported for this task"))

    def get_label_from_image(self, image):
        subject_id, type_id, sample_id = extract_subject_id_image_type_and_second_id(image)
        if self.task_id == "1":
            return type_id
        elif self.task_id == "2":
            return int(subject_id)
        elif self.task_id == "3":
            return int(sample_id)

        raise(Exception("Unable to fetch label id for image: {}".format(image)))

    def calculate_misses_and_false_positives(self, results):
        misses = list()
        false_positives = list()
        test_images = list(self.test_data.keys())
        predicted_data = {test_images[index]: result for index, result in enumerate(results)}
        original_data = {image: self.get_label_from_image(image) for image in test_images}
        print("original data:: {}".format(original_data))
        print("predicetd data:: {}".format(predicted_data))
        for image, label in predicted_data.items():
            original_label = original_data.get(image, None)
            if original_label != label:
                false_positives.append(image)
        # print("length: {} ==== false positivess:: {}".format(len(false_positives), false_positives))

        return false_positives, None
