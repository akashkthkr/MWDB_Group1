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
        for original_category_id in self.get_labels():
            count = count + 1
            training_image_ids, training_set_X, training_set_Y = [], [], []
            for image_id in self.train_data:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                training_image_ids.append(image_id)
                training_set_X.append(self.train_data[image_id])
                if type_id == original_category_id:
                    training_set_Y.append(1)
                else:
                    training_set_Y.append(0)

            test_image_ids, test_set_X = [], []

            for image_id in self.test_data:
                test_image_ids.append(image_id)
                test_set_X.append(self.test_data[image_id])
            clf.fit(training_set_X, training_set_Y)
            prediction_values = clf.predict(numpy.array(test_set_X))
            predictions = numpy.sign(prediction_values)

            index = 0
            for image_id in test_image_ids:
                label_name_by_image = self.get_category(image_id)
                if predictions[index] == 1:
                    if original_category_id == label_name_by_image:
                        if original_category_id in self.true_positives:
                            self.true_positives[original_category_id].append(image_id)
                        else:
                            true_positive_list = [image_id]
                            if len(self.true_positives) == 0:
                                self.true_positives = {original_category_id: true_positive_list}
                            else:
                                self.true_positives[original_category_id] = true_positive_list
                    else:
                        if original_category_id in self.false_positives:
                            self.false_positives[original_category_id].append(image_id)
                        else:
                            false_positive_list = [image_id]
                            if len(self.false_positives) == 0:
                                self.false_positives = {original_category_id: false_positive_list}
                            else:
                                self.false_positives[original_category_id] = false_positive_list

                    if image_id in images_assosciation:
                        if any(original_category_id in type for type in images_assosciation[image_id]):
                            for value in images_assosciation[image_id]:
                                value[original_category_id] = max(value[original_category_id], prediction_values[index])
                        else:
                            y = {original_category_id: prediction_values[index]}
                            images_assosciation[image_id].append(y)
                    else:
                        x = {original_category_id: prediction_values[index]}
                        images_assosciation[image_id] = [x]
                else:
                    if original_category_id == label_name_by_image:
                        if original_category_id in self.false_negatives:
                            self.false_negatives[original_category_id].append(image_id)
                        else:
                            false_negative_list = [image_id]
                            if len(self.false_negatives) == 0:
                                self.false_negatives = {original_category_id: false_negative_list}
                            else:
                                self.false_negatives[original_category_id] = false_negative_list
                    else:
                        if original_category_id in self.true_negatives:
                            self.true_negatives[original_category_id].append(image_id)
                        else:
                            true_negative_list = [image_id]
                            if len(self.true_negatives) == 0:
                                self.true_negatives = {original_category_id: true_negative_list}
                            else:
                                self.true_negatives[original_category_id] = true_negative_list

                    if image_id in negative_image_assosciation:
                        if any(original_category_id in type for type in negative_image_assosciation[image_id]):
                            for value in negative_image_assosciation[image_id]:
                                value[original_category_id] = max(value[original_category_id], prediction_values[index])
                        else:
                            y = {original_category_id: prediction_values[index]}
                            negative_image_assosciation[image_id].append(y)
                    else:
                        x = {original_category_id: prediction_values[index]}
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

        dt_utils = DTUtilities(self.false_positives, self.true_positives, self.false_negatives,
                                     self.true_negatives, self.get_labels(), self.task_id, self.test_data, classifier_results)
        dt_utils.save_fp_misses_rate()

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
