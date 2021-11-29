import Phase3.Lsh
import numpy as np
from termcolor import colored
from Phase3.Lsh import Lsh

'''Running the code:
$ python lsh.py
    
    Enter the ImageId: <image_id>
    Enter the number of Layers: <number> l
    Enter the number of Hashes per layer: <number> k
    Enter the value of t: <task_number>
'''


def lsh_executor(train_features, test_features):
    num_layers = int(input("Number of Layers="))
    num_hash_function = int(input("Enter the number of Hashes per layer="))
    t = int(input("Number of nearest neighbors required to query image="))
    image_vectors = []
    image_ids = []
    first_key = next(iter(train_features))
    lsh = Lsh(number_of_hashes_per_layer=num_hash_function, number_of_features=len(train_features[first_key]),
              num_layers=num_layers)
    for image_id in train_features:
        image_vectors.append(train_features[image_id])
        image_ids.append(image_id)
        lsh.add_to_index_structure(input_feature=train_features[image_id], image_id=image_id)
    image_vectors = np.asarray(image_vectors)
    print(colored("num features per image=" + str(len(image_vectors[0])), 'blue'))
    query_ids = []
    query_vector = []
    for query_id in test_features:
        query_ids.append(query_id)
        query_vector.append(test_features[query_id])

    query_vector = np.array(query_vector[0])

    ret_result, overall_image_considered, unique_img, bucket_size, size_of_index_structure = lsh.query(feature=query_vector, num_results=t)
    print(colored("Query Image:" + str(query_ids[0]), 'green'))
    print(colored("Size of the index structure created=" + str(size_of_index_structure), 'green'))
    print(colored("No of buckets Searched=" + str(bucket_size), 'green'))
    print(colored("Total no of images considered=" + str(overall_image_considered), 'green'))
    print(colored("Unique Images considered=" + str(unique_img), 'green'))

    return ret_result
