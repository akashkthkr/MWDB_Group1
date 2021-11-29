import numpy
import json
from numpyencoder import NumpyEncoder
from Phase3.decision_tree.decision_tree import DecisionTree
from constants.Constants_Phase3 import OUTPUTS_PATH
from Phase3.decision_tree.utils import DTUtilities

from Project.services.ImageFetchService import extract_subject_id_image_type_and_second_id


TASK_LABEL_MAPPING = {
    1: "X labels",
    2: "Y labels",
    3: "Z labels"
}


class DecisionTreeHandler2:
    def __init__(self, task_id=None, train_data=None, test_data=None):
        self.task_id = task_id
        self.train_data = train_data
        self.test_set_features = test_data
        self.labels = self.get_labels()
        self.false_positives, self.false_negatives, self.true_positives, self.true_negatives = {}, {}, {}, {}
        self.subject_ids_list = self.get_labels()

    def execute(self):
        images_assosciation = {}
        negative_image_assosciation = {}
        clf = DecisionTree()
        count = 1
        for original_subject_id in self.subject_ids_list:
            count = count + 1
            training_image_ids, training_set_X, training_set_Y = [], [], []
            for image_id in self.train_data:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                training_image_ids.append(image_id)
                training_set_X.append(self.train_data[image_id])
                if subject_id == original_subject_id:
                    training_set_Y.append(1)
                else:
                    training_set_Y.append(0)

            test_image_ids, test_set_X = [], []

            for image_id in self.test_set_features:
                test_image_ids.append(image_id)
                test_set_X.append(self.test_set_features[image_id])
            clf.fit(training_set_X, training_set_Y)
            prediction_values = clf.predict(numpy.array(test_set_X))
            predictions = numpy.sign(prediction_values)

            index = 0
            for image_id in test_image_ids:
                subject_id, type_id, image_sample_id = extract_subject_id_image_type_and_second_id(image_id)
                if predictions[index] == 1:
                    if original_subject_id == subject_id:
                        if original_subject_id in self.true_positives:
                            self.true_positives[original_subject_id].append(image_id)
                        else:
                            true_positive_list = [image_id]
                            if len(self.true_positives) == 0:
                                self.true_positives = {original_subject_id: true_positive_list}
                            else:
                                self.true_positives[original_subject_id] = true_positive_list
                    else:
                        if original_subject_id in self.false_positives:
                            self.false_positives[original_subject_id].append(image_id)
                        else:
                            false_positive_list = [image_id]
                            if len(self.false_positives) == 0:
                                self.false_positives = {original_subject_id: false_positive_list}
                            else:
                                self.false_positives[original_subject_id] = false_positive_list

                    if image_id in images_assosciation:
                        if any(original_subject_id in type for type in images_assosciation[image_id]):
                            for value in images_assosciation[image_id]:
                                value[original_subject_id] = max(value[original_subject_id], prediction_values[index])
                        else:
                            y = {original_subject_id: prediction_values[index]}
                            images_assosciation[image_id].append(y)
                    else:
                        x = {original_subject_id: prediction_values[index]}
                        images_assosciation[image_id] = [x]
                else:
                    if original_subject_id == subject_id:
                        if original_subject_id in self.false_negatives:
                            self.false_negatives[original_subject_id].append(image_id)
                        else:
                            false_negative_list = [image_id]
                            if len(self.false_negatives) == 0:
                                self.false_negatives = {original_subject_id: false_negative_list}
                            else:
                                self.false_negatives[original_subject_id] = false_negative_list
                    else:
                        if original_subject_id in self.true_negatives:
                            self.true_negatives[original_subject_id].append(image_id)
                        else:
                            true_negative_list = [image_id]
                            if len(self.true_negatives) == 0:
                                self.true_negatives = {original_subject_id: true_negative_list}
                            else:
                                self.true_negatives[original_subject_id] = true_negative_list

                    if image_id in negative_image_assosciation:
                        if any(original_subject_id in type for type in negative_image_assosciation[image_id]):
                            for value in negative_image_assosciation[image_id]:
                                value[original_subject_id] = max(value[original_subject_id], prediction_values[index])
                        else:
                            y = {original_subject_id: prediction_values[index]}
                            negative_image_assosciation[image_id].append(y)
                    else:
                        x = {original_subject_id: prediction_values[index]}
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

        for image_id in self.test_set_features:
            if image_id not in classifier_results:
                remaining_images.append(image_id)
                ans = -1000000000
                for X in negative_image_assosciation[image_id]:
                    for key in X:
                        if X[key] > ans:
                            classifier_results[image_id] = key
                            ans = X[key]

        jsonString = json.dumps(images_assosciation, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + "DT_Classifier_Values_Task2.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        jsonString = json.dumps(negative_image_assosciation, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + "DT_Classifier_negative_Values_Task2.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        jsonString = json.dumps(classifier_results, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + "DT_Classifier_Results_Task2.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        svm_utilities = DTUtilities(self.false_positives, self.true_positives, self.false_negatives,
                                     self.true_negatives, self.subject_ids_list, self.task_id, self.test_set_features,
                                     classifier_results)
        svm_utilities.save_fp_misses_rate()
        print(remaining_images)

    def get_category(self, image_id):
        subject_id, type_id, image_sample_id = extract_subject_id_image_type_and_second_id(image_id)
        if self.task_id == "1":
            return type_id
        elif self.task_id == "2":
            return subject_id
        elif self.task_id == "3":
            return image_sample_id
        pass

    def get_labels(self):
        if self.task_id == "1":
            return ["cc", "con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot",
                    "smooth", "stipple"]
        elif self.task_id == "2":
            return [str(i) for i in range(1, 41, 1)]
        elif self.task_id == "3":
            return [str(i) for i in range(1, 11, 1)]
        else:
            raise(Exception("Decision Tree will not be supported for this task"))
