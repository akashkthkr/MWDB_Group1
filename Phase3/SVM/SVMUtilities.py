import json

from numpyencoder import NumpyEncoder

from constants.Constants_Phase3 import OUTPUTS_PATH


class SVMUtilities():

    def __init__(self, false_positives=None, true_positives=None, false_negatives=None, true_negatives=None, tag_ids=None, task_id = None):
        self.false_positives = false_positives
        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.true_negatives = true_negatives
        self.tag_ids = tag_ids
        self.false_positive_rate, self.misses_rate = {}, {}
        self.task_id = task_id

    def save_fp_misses_rate(self):
        for tag_id in self.tag_ids:
            if tag_id in self.false_positives:
                fp_count = float(len(self.false_positives[tag_id]))
            else:
                fp_count = 0
            if tag_id in self.true_positives:
                tp_count = float(len(self.true_positives[tag_id]))
            else:
                tp_count = 0

            if tag_id in self.false_negatives:
                fn_count = float(len(self.false_negatives[tag_id]))
            else:
                fn_count = 0
            if tag_id in self.true_negatives:
                tn_count = float(len(self.true_negatives[tag_id]))
            else:
                tn_count = 0

            if fp_count == 0 and tn_count == 0:
                self.false_positive_rate[tag_id] = 0
            else:
                self.false_positive_rate[tag_id] = fp_count/(fp_count + tn_count)


            if fn_count == 0 and tp_count == 0:
                self.misses_rate[tag_id] = 0
            else:
                self.misses_rate[tag_id] = fn_count / (fn_count + tp_count)


        self.save_file(self.false_positive_rate, f"SVM_Classifier_fpr_Task{self.task_id}.json")
        self.save_file(self.misses_rate, f"SVM_Classifier_mr_Task{self.task_id}.json")
        self.save_file(self.false_positives, f"SVM_Classifier_fp_Task{self.task_id}.json")
        self.save_file(self.true_positives, f"SVM_Classifier_tp_Task{self.task_id}.json")
        self.save_file(self.false_negatives, f"SVM_Classifier_misses_Task{self.task_id}.json")
        self.save_file(self.true_negatives, f"SVM_Classifier_tn_Task{self.task_id}.json")

    def save_file(self, json_content, file_name):
        jsonString = json.dumps(json_content, cls=NumpyEncoder)
        jsonFile = open(OUTPUTS_PATH + file_name, "w")
        jsonFile.write(jsonString)
        jsonFile.close()