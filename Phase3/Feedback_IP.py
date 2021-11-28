import Phase3.Feedback as Feedback
from Phase3.SVM.SVM import SupportVectorMachine
import collections

def execute_flow(ip_features,query_feature,result_ids):
    r = int(input('Number of images  to label as relevant:'))
    ir = int(input('Number of images to label as irrelevant:'))
    k = len(result_ids)
    print(k)
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
        #dt = DT(10)
        #dt.fit(fb.X,fb.Y)
        #for id,feature in ip_features:
        #    Labelled_Dataset[id] = dt.predict(feature)


    elif classifier == 'SVM':
        print("Classify entire database with relevant/irrelevant feedback")
        Svm = SupportVectorMachine()
        Svm.fit(fb.X,fb.Y)
        for id,feature in ip_features.items():
            Labelled_Dataset[id] = Svm.predict(feature)



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
    for val in new_ordered_images[:k]:
        result[val[0]] = val[1]
        print(val[0],val[1])

    #TODO:Save Result
    fb.save_result(result)




