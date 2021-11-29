import numpy as np

import Phase3.Feedback as Feedback
from Phase3.SVM.SVM import SupportVectorMachine,gaussian_kernel
from Phase3.decision_tree import decision_tree
from Project.services.ResultSubmission import  write_result
import collections

def execute_flow(ip_features,query_feature,result_ids):

    k = len(result_ids)
    #path = "C:\\Users\\saiac\\Documents\\MWDB-Phase3\\MWDB_Group1\\inputs\\100\\"
    #write_result(path,result_ids)
    fb = Feedback.Feedback()
    base_img = []
    for query_id in query_feature:
        base_img = query_feature[query_id]
    num_image = {}
    count = 1
    rorir_map = {}
    #Label all images -2 , then label relevant images as 1 and irrelevant images as 0
    for image_id in result_ids:
        #image_id = os.path.basename(image_id)
        num_image[count] = image_id
        print(count, image_id)
        rorir_map[num_image[count]] = -2
        count += 1

    r = int(input('Number of images  to label as relevant:'))
    ir = int(input('Number of images to label as irrelevant:'))

   # if r == 0:
   #     print("Edge case:Peforming unweighted knn since there are no relevant images in the results")
   #     #features = [feature for _,feature in ip_features.items()]
   #     knn_total = {image_id:fb.euclidean_distance(base_img, ip_features[image_id]) for
   #                           image_id,_ in ip_features.items()}
   #     fb.save_result(knn_total)
   #     return

    while r > 0:
        ind = int(input('Enter the image id to label as Relevant:'))
        r -= 1
        #if count == 0:
            #base_id = num_image[ind]
            #count += 1
        rorir_map[num_image[ind]] = 1
    while ir > 0:
        ind = int(input('Enter the image id to label as irrelevant:'))
        ir -= 1
        rorir_map[num_image[ind]] = -1

    fb.generate_input_data(rorir_map,ip_features)
    classifier = input("1.SVM\n2.DT\nSelect Classifier: ")

    Labelled_Dataset = {}

    if classifier == 'DT':
    # TODO: use generated input data and relabel the images
        print("Classify entire database with relevant/irrelevant feedback")
        dt = decision_tree.DecisionTree()
        X_list = fb.X.tolist()
        dt.fit(X_list,fb.Y)
        features = []
        ids = []
        for id,feature in ip_features.items():
            features.append(feature)
            ids.append(id)
        predicted_labels = dt.predict(features)
        Labelled_Dataset = {ids[i]:predicted_labels[i] for i in range(len(ids))}


    elif classifier == 'SVM':
        print("Classify entire database with relevant/irrelevant feedback")
        Svm = SupportVectorMachine(gaussian_kernel, C=500)
        Svm.fit(fb.X,fb.Y)
        features = []
        ids = []
        for id, feature in ip_features.items():
            features.append(feature)
            ids.append(id)
        predicted_labels = Svm.predict(features)
        Labelled_Dataset = {ids[i]: predicted_labels[i] for i in range(len(ids))}



    # TODO: use generated input data and relabel the images
    #old_list_images = list()
    #for image_id, val in rorir_map.items():
    #    old_list_images.append(image_id)
    relevant_image_ids = []
    for id,label in Labelled_Dataset.items():
        if label == 1:
            relevant_image_ids.append(id)
    #relevant_images = [(image_id,feature) for id,feature in Labelled_Dataset.items()]
    #Unweighted knn on relevant images to get the new results
    new_ordered_images = [(image_id,
                           fb.euclidean_distance(base_img, ip_features[image_id])) for
                          image_id in relevant_image_ids]#old_list_images]
    new_ordered_images.sort(key=lambda v: v[1])

    #Ordered dict
    result = collections.OrderedDict()
    resultids = []
    for val in new_ordered_images[:k]:
        result[val[0]] = val[1]
        print(val[0],val[1])
        resultids.append(val[0])


    #write_result(path,resultids)

    #TODO:Save Result
    fb.save_result(result)




