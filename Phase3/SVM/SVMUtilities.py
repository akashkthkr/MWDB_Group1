import json

from numpyencoder import NumpyEncoder

from Project.services.ImageFetchService import extract_subject_id_image_type_and_second_id
from constants.Constants_Phase3 import OUTPUTS_PATH


class SVMUtilities():

    def __init__(self, false_positives=None, true_positives=None, false_negatives=None, true_negatives=None, tag_ids=None, task_id = None, features = None, results = None):
        self.false_positives = false_positives
        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.true_negatives = true_negatives
        self.tag_ids = tag_ids
        self.false_positive_rate, self.misses_rate = {}, {}
        self.task_id = task_id
        self. features = features
        self.results = results

    def save_fp_misses_rate(self):
        image_ids = self.extract_ids_from_images(self.features, self.task_id)
        result_type_ids = self.extract_ids_from_results(self.results)
        true_positives, false_positives = self.extract_true_false_positives(result_type_ids, self.task_id)
        misses = self.extract_misses(image_ids, true_positives)
        for tag_id in self.tag_ids:
            if tag_id in false_positives:
                fp_count = float(len(false_positives[tag_id]))
            else:
                fp_count = 0
            if tag_id in true_positives:
                tp_count = float(len(true_positives[tag_id]))
            else:
                tp_count = 0

            if tag_id in misses:
                fn_count = float(len(misses[tag_id]))
            else:
                fn_count = 0
            # if tag_id in self.true_negatives:
            #     tn_count = float(len(self.true_negatives[tag_id]))
            # else:
            #     tn_count = 0

            if fp_count == 0:
                self.false_positive_rate[tag_id] = 0
            else:
                self.false_positive_rate[tag_id] = fp_count/(float(len(self.features)))


            if fn_count == 0 and tp_count == 0:
                self.misses_rate[tag_id] = 0
            else:
                self.misses_rate[tag_id] = fn_count / (fn_count + tp_count)


        self.save_file(self.false_positive_rate, f"SVM_Classifier_fpr_Task{self.task_id}.json")
        self.save_file(self.misses_rate, f"SVM_Classifier_mr_Task{self.task_id}.json")
        self.save_file(false_positives, f"SVM_Classifier_fp_Task{self.task_id}.json")
        self.save_file(true_positives, f"SVM_Classifier_tp_Task{self.task_id}.json")
        self.save_file(misses, f"SVM_Classifier_misses_Task{self.task_id}.json")
        # self.save_file(self.true_negatives, f"SVM_Classifier_tn_Task{self.task_id}.json")

    def save_file(self, json_content, file_name):
        jsonString = json.dumps(json_content, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + file_name, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def extract_ids_from_images(self, features, task_id):
        image_ids_dict = {}
        if task_id == "1":
            for image_id in features:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                if type_id in image_ids_dict:
                    image_ids_dict[type_id].append(image_id)
                else:
                    image_ids_dict[type_id] = [image_id]
        elif task_id == "2":
            for image_id in features:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                if subject_id in image_ids_dict:
                    image_ids_dict[subject_id].append(image_id)
                else:
                    image_ids_dict[subject_id] = [image_id]
        else:
            for image_id in features:
                subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                if second_id in image_ids_dict:
                    image_ids_dict[second_id].append(image_id)
                else:
                    image_ids_dict[second_id] = [image_id]

        return image_ids_dict

    def extract_ids_from_results(self, results):
        results_dict = {}
        for image_id in results:
            if results[image_id] in results_dict:
                results_dict[results[image_id]].append(image_id)
            else:
                results_dict[results[image_id]] = [image_id]
        return results_dict

    def add_true_false_positives(self, true_positives, false_positives, image_id, tag_id, type_id):

        if type_id == tag_id:
            if tag_id in true_positives:
                true_positives[tag_id].append(image_id)
            else:
                true_positives[tag_id] = [image_id]
        else:
            if tag_id in false_positives:
                false_positives[tag_id].append(image_id)
            else:
                false_positives[tag_id] = [image_id]

    def extract_true_false_positives(self, result, task_id):
        true_positives, false_positives = {}, {}
        if task_id == "1":
            for tag_id in result:
                for image_id in result[tag_id]:
                    subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                    self.add_true_false_positives(true_positives, false_positives, image_id, tag_id, type_id)
        elif task_id == "2":
            for tag_id in result:
                for image_id in result[tag_id]:
                    subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                    self.add_true_false_positives(true_positives, false_positives, image_id, tag_id, subject_id)
        else:
            for tag_id in result:
                for image_id in result[tag_id]:
                    subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
                    self.add_true_false_positives(true_positives, false_positives, image_id, tag_id, second_id)
        return true_positives, false_positives

    def extract_misses(self, image_ids, true_positives):
        misses = {}
        for tag_id in image_ids:
            if tag_id in true_positives:
                if len(image_ids[tag_id]) > len(true_positives[tag_id]):
                    misses[tag_id] = list(set(image_ids[tag_id]) - set(true_positives[tag_id]))
            else:
                misses[tag_id] = image_ids[tag_id]

        return misses
