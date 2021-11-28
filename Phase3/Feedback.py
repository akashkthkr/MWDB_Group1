import os
import sys
from pathlib import Path
import json
import numpy as np
from constants.Constants_Phase3 import OUTPUTS_PATH
from Phase3.vafiles import save_to_json

class Feedback:
    def __init__(self):
        self.result = None
        self.result_path = OUTPUTS_PATH #Path to store new result
        #self.dataset = list()
        self.X = None
        self.Y = None

    #def set_result(self):
     #   print("Getting result from previous tasks")
     #   self.result = {1:"X",2:"X",3:"X",4:"X",5:"X",6:"Y",7:"Y",8:"Y",9:"Y",10:"Y"}
        #self.result = misc.load_from_pickle(self.reduced_pickle_file_folder, 'Task_5_Result')

    def generate_input_data(self, rorir_map, dataset_features):
        X = []
        Y = []

        for image_id, label in rorir_map.items():
            #image_id = os.path.basename(image_id)
            if label == -1 or label == 1:
                X.append(dataset_features[image_id])
                Y+=[rorir_map[image_id]]
        X = np.array(X)
        Y = np.array(Y)
        self.X=X
        self.Y=Y

        return

    def euclidean_distance(self, dist1, dist2):
        return (sum([(a - b) ** 2 for a, b in zip(dist1, dist2)])) ** 0.5

    def save_result(self, result):
        save_to_json(self.result_path+"feedback.json",result)
