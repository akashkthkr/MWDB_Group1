import glob
import ntpath
import json
import numpy as np

from skimage.io import imread
from numpyencoder import NumpyEncoder
from Project.Lda.code import constants_lda
from Lda.code.features import get_all_features
from constants import Constants_phase2 as Constants_p2



# Function to load the images with image ids from the inputs folder
from utilities import GreyScaleNormalization


def load_dataset_from_folder():
    dataset = []
    folder_path = Constants_p2.PATH + "\\*.png"
    for path in glob.glob(folder_path):
        id = ntpath.splitext(ntpath.basename(path))[0]
        image = imread(path)
        image = np.array(image).astype(np.float)
        GreyScaleNormalization.get_normalized_image(image)
        dataset.append({'id': id, 'image': image})
    return dataset

def load_dataset_using_Y(Y):
    dataset = []
    folder_path = Constants_p2.PATH + "\\" + constants_lda.IMAGE + "-*-" + Y + "-*.png"
    for path in glob.glob(folder_path):
        id = ntpath.splitext(ntpath.basename(path))[0]
        image = imread(path)
        image = np.array(image).astype(np.float)
        GreyScaleNormalization.get_normalized_image(image)
        dataset.append({'id': id, 'image': image})
    return dataset

def load_dataset_using_X(X):
    dataset = []
    folder_path = Constants_p2.PATH + "\\" + constants_lda.IMAGE +'-' + X +"-*.png"
    for path in glob.glob(folder_path):
        id = ntpath.splitext(ntpath.basename(path))[0]
        image = imread(path)
        image = np.array(image).astype(np.float)
        GreyScaleNormalization.get_normalized_image(image)
        dataset.append({'id': id, 'image': image})
    return dataset

# Saves JSON file the given directory
def save_json(data, filename):
    with open(filename, "w") as file:
        jsonString = json.dumps(data, cls=NumpyEncoder)
        file.write(jsonString)

# Opens JSON file from given directory
def open_json(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.readlines()[0])
    return data

# Procesess data after recieving from dataset. Fetches features from the functions
def process_data(task='all', X='con', Y='1'):
    print("In process data method")
    dataset = []
    # dataset = load_dataset_from_folder()
    if task == 'all':
        dataset = load_dataset_from_folder()

    if task == 'task1':
        dataset = load_dataset_using_X(X)

    if task == "task2":
        dataset = load_dataset_using_Y(Y)

    data = {}

    for file in dataset:
        id = file['id']
        features = get_all_features(file['image'])
        cm = features['cm']
        elbp = features['elbp']
        hog = features['hog']
        combined = features['combined']
        payload = {
            "image": file['image'],
            "features": {
                "cm": cm,
                "elbp": elbp,
                "hog": hog,
                "combined": combined
            }
        }

        data[id] = payload

    filename = "D:\\Masters\\CSE 515\\Project\\Lda\\code\\"+constants_lda.FOLDER +"_"+ task + "_database.json"
    print("Saving:"+ filename)
    save_json(data, filename)

    print("Data Processed.")
    return filename
