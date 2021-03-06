import json
import logging
import math
import pickle
import sys
from termcolor import colored
import numpy as np

from constants.Constants_Phase3 import OUTPUTS_PATH

ARBITRARY_CONSTANT = 5
ARBITRARY_SMALL_CONSTANT = 0.000000000001

def load_vectors(filename_pk):
    with open(filename_pk, 'rb') as pickle_file:
        transformed_data = pickle.load(pickle_file)
    return transformed_data

#get lower and upper bounds
def get_bounds(query_vector, partitions, query_approximation, vector_approximation, p_in_lp):
    b = len(query_approximation)/len(query_vector)
    d = len(query_vector)
    start_index = 0
    lij = []
    uij = []
    for j,vqj in enumerate(query_vector):
        pj = partitions[j]
        #number of bits in this dimension
        bj = int(b)
        assert math.ceil(bj)==math.floor(bj)
        #bj = math.floor(b / d) + 1 if j < b % d else math.floor(b / d)
        rqj_str = query_approximation[start_index:start_index+bj]
        rij_str = vector_approximation[start_index:start_index+bj]
        #converting binary string to integer for indexing of pj
        rqj = int(rqj_str,2)
        rij = int(rij_str,2)
        if rij < rqj:
            lij.append(vqj-pj[rij+1])
            uij.append(vqj-pj[rij])
        elif rij == rqj:
            lij.append(0)
            uij.append(max(vqj-pj[rij],pj[rij+1]-vqj))
        else:
            lij.append(pj[rij]-vqj)
            uij.append(pj[rij+1]-vqj)
        start_index+=bj

    lij = np.array(lij)
    uij = np.array(uij)
    # distance calculation - lower and upper bounds
    lij = np.power(lij,p_in_lp)
    uij = np.power(uij,p_in_lp)
    li = math.pow(np.sum(lij),1.0/p_in_lp)
    ui = math.pow(np.sum(uij),1.0/p_in_lp)

    return li,ui

# num_partition_points = num_partitions+1
# partitions created by principle of equally filled regions
def get_partition_points(vector, num_partitions):

    # number of points have to be atleast the number of partitions.
    # if this condition not satisfied, then arbitrary empty partitions may exist which we do not want.
    # Hence, we enforce the condition and throw exception if this condition not met
    if len(vector)<num_partitions:
        print(colored("number of points "+str(len(vector))+" less than the number of partitions "+
                      str(int(num_partitions))+". Please choose suitable number of bit representation and/or dimensions",
                      "red"))
        assert len(vector)>=num_partitions
    num_points = len(vector)

    # some partitions have greater number of points and some lesser if number of points not divisible by number of partitions
    # initial partitions given larger number of points (arbitrary decision)
    num_points_per_partition_list = np.array([math.floor(num_points/num_partitions)+1
                                              if i<num_points%num_partitions else math.floor(num_points/num_partitions)
                                              for i in range(int(num_partitions))])
    print("num_points_per_partition=",num_points_per_partition_list)

    # first partition point is min-5, last partition point is max value in vector + 5
    # ARBIRARARY_CONSTANT added so that the last point is not at the boundary and is atleast within a certain distance of the partition boundary
    # partition point is kept as mid point of last point of nth partition and first point of n+1th partition
    # running sum of number of points per partition maintained to index points in these partitions
    vsorted = sorted(vector)
    mn = min(vector) - ARBITRARY_CONSTANT
    mx = max(vector) + ARBITRARY_CONSTANT
    partition_points = []
    partition_points.append(mn)
    point_count = 0
    #last index skipped as not required as last partition point = max_value+ARBIRARARY_CONSTANT
    for num_points in num_points_per_partition_list[:-1]:
        point_count+=num_points
        mid_point = (vsorted[point_count-1]+vsorted[point_count])/2
        partition_points.append(mid_point)
    #partition_points.append(mx)

    #trying large value for last boundary...replace with mx=ARBIRARY_CONSTANT later if required
    last_ele = partition_points[-1]
    partition_points.append(last_ele+ARBITRARY_CONSTANT)

    return partition_points

def get_binary_string(num, length):
    b = bin(num)[2:]
    orig_len = len(b)
    padding = '0'*(length-orig_len)
    return padding+b

def get_partitions(matrix,b):
    partition = []
    d = matrix.shape[1]
    for j in range(d):
        print("j=",j)
        # b is number of bits per dimension and not total number of bits for approximation of vector
        bj = b
        #bj = math.floor(b/d)+1 if j<b%d else math.floor(b/d)
        print("bj=",bj)
        vj = matrix[:,j]
        pj = get_partition_points(vj,math.pow(2,bj))
        print("pj=", pj)
        partition.append(pj)
    return partition

def create_approximation(vectors, partitions):
    approximations = []
    vector_len_set = set()
    bin_string_length = set()
    for vector in vectors:
        approximation_vector = ''
        for j,vj in enumerate(vector):
            vector_len_set.add(len(vector))
            if len(vector_len_set)!=1:
                print("len_vectors=",vector_len_set)
            pj = partitions[j]
            #thresholding to make sure that the value of vector doesn't exceed the last partition. large number will be in the last partition of each dimension
            vj = min(vj,pj[-1]-ARBITRARY_SMALL_CONSTANT)
            num_bits_in_partition = int(math.log(len(pj)-1,2))
            found = False
            for i,val in enumerate(pj[:-1]):
                if vj<val:
                    print(colored("invalid vj --less",'red'))
                if vj>pj[-1]:
                    print(colored("invalid vj --more", 'red'))
                if vj>=val and vj<pj[i+1]:
                    bin_str = get_binary_string(i,num_bits_in_partition)
                    bin_string_length.add(len(bin_str))
                    if len(bin_string_length)!=1:
                        print("INVALID binary strings with unequal length created:",bin_string_length)
                        exit(1)
                    approximation_vector+=bin_str
                    found = True
                    break
            if found==False:
                print("found bucket=",found,"vj=",vj,"pj=",pj)
                exit(1)
        approximations.append(approximation_vector)
    print("len_approximation=", len(approximations[0]))
    return approximations

def save_to_json(filename, object_to_store):
    with open(filename, 'w') as json_file:
        json.dump(object_to_store,json_file)

def save_to_pickle(filename, object_to_store):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(object_to_store,pickle_file)

def create_and_save_va_file(vectors, b, vector_ids, output_folder):
    partitions = get_partitions(vectors,b)
    print(colored("partitions=",'blue'))
    print(colored(partitions,'blue'))
    approximations = create_approximation(vectors,partitions)
    # size computations for deliverables
    vafile_size = sys.getsizeof(approximations)
    vafile_metadata_size = sys.getsizeof(partitions)
    original_size = sys.getsizeof(vectors)
    compression_ratio = original_size/(vafile_size+vafile_metadata_size)

    # create dicts so that json serializable
    app = {'approximations':approximations, 'vector_ids':vector_ids}

    vafile_stats = {"va_file_size":vafile_size,
                       "meta_data_size":vafile_metadata_size,
                        "original_vectors_size=":original_size,
                       "compression_ratio":compression_ratio}
    num_dim = len(vectors[0])
    approximation_len = len(approximations[0])
    output_filename = "vafile_stats_"+str(num_dim)+"_"+str(approximation_len)+".json"
    print(colored(vafile_stats,'green'))
    save_to_json(output_folder+"va_file.json",app)
    save_to_pickle(output_folder+"partitions.pk",partitions)
    save_to_pickle(output_folder+"vectors.pk",vectors)
    save_to_json(output_folder+output_filename,vafile_stats)

def l_norm_similarity(vector1:np.ndarray, vector2:np.ndarray, li=2):
    assert vector1.shape == vector2.shape and vector1.ndim == 1 and li >= 1
    sm = np.sum(np.power(np.abs(vector1-vector2),li))
    return math.pow(sm, 1/float(li))

class Candidate_VA_SSA:
    def __init__(self,k):
        self.dst = np.full((k), np.inf,dtype=float)
        self.ans = np.full((k),-1,dtype=int)
        self.k = k

    def init_candidate(self):
        self.dst = np.full((self.k), np.inf,dtype=float)
        self.ans = np.full((self.k),-1,dtype=int)
        return self.dst[self.k-1]

    def candidate(self,d,i):
        if d < self.dst[self.k-1]:
            self.dst[self.k-1] = d
            self.ans[self.k-1] = i
            s = sorted(zip(self.dst,self.ans))
            s = np.array(s)
            self.dst = np.asarray(s[:,0],dtype=float)
            self.ans = np.asarray(s[:,1],dtype=int)
        return self.dst[self.k-1]

    def get_indexes(self):
        return self.ans


def va_ssa(va, v, qa, q, k, partitions, v_ids, p):
    buckets_searched = set()
    total_images_in_database = len(v)
    num_buckets = 0
    #for partition in partitions:
    try:
        num_buckets = math.pow(len(partitions[0])-1, len(partitions))
    except OverflowError:
        num_buckets = float('inf')

    num_images_considered = 0
    buckets_having_elements = set()
    candidate_obj = Candidate_VA_SSA(k)
    d = candidate_obj.init_candidate()
    for i, vai in enumerate(va):
        buckets_having_elements.add(vai)
        li, _ = get_bounds(q, partitions, qa, vai, p)
        #buckets_searched.add(li)
        if li < d:
            buckets_searched.add(vai)
            num_images_considered += 1
            lvivq = l_norm_similarity(np.array(v[i]), np.array(q))
            d = candidate_obj.candidate(lvivq, i)
    top_k_indexes = candidate_obj.get_indexes()
    num_buckets_searched = len(buckets_searched)
    comparisons_saved = total_images_in_database-num_images_considered
    percentage_compairsons_saved = comparisons_saved/total_images_in_database*100
    print(colored('Total image comparisons saved with va files:'+str(comparisons_saved)+
                  "--- i.e. "+str(percentage_compairsons_saved)+"% images not visited",'blue'))
    deliverable = {"total images in database":total_images_in_database,
                   "number of buckets":num_buckets,
                   "num of buckets having images":len(buckets_having_elements),
                   "number of buckets searched":num_buckets_searched,
                   "number of images visited":num_images_considered,
                   "number of unique images visited":num_images_considered,
                   "percentage comparisons saved":percentage_compairsons_saved}
    print(colored(deliverable,'green'))

    num_dim = len(q)
    approximation_len = len(qa)
    output_filename = "va_query_stats"+"_"+str(num_dim)+"_"+str(approximation_len)+".json"
    save_to_json(OUTPUTS_PATH+output_filename,deliverable)

    v_ids = np.array(v_ids,dtype=str)
    return v_ids[top_k_indexes], num_images_considered

def va_noa(va, v, qa, q, k, v_ids):
    pass

def get_n_closest_images(query_vector, images_vectors, n, image_ids, p_in_lp):
    distances = np.array([l_norm_similarity(np.array(query_vector),np.array(image_vector), p_in_lp) for image_vector in images_vectors])
    top_features_ind = distances.argsort()[:n]
    print("top_features_ind=",top_features_ind)
    top_images = [image_ids[ind] for ind in top_features_ind]
    print("topn ids=",top_features_ind," top_n_images=",top_images)
    return top_images

def calculate_statistics(top_k_knn, top_k_va, num_images_considered):
    correct_images = list(set(top_k_va).intersection(top_k_knn))
    print(colored("correct_knn_va="+str(correct_images),'blue'))
    miss_rate = (len(top_k_knn) - len(correct_images))/len(top_k_knn)*100
    misses = list(set(top_k_knn) - set(top_k_va))
    print(colored("misses="+str(misses),'blue'))
    print(colored("miss_rate="+str(miss_rate)+'%','green'))

    false_positive_rate = (num_images_considered-len(correct_images))/num_images_considered*100
    print(colored("false_positive_rate="+str(false_positive_rate)+'%','green'))
    return miss_rate,false_positive_rate


def va_search(input_folder, query_vector, k, algorithm='va_ssa',p_in_lp=1):
    with open(input_folder+"\\va_file.json", 'rb') as json_file:
        va_file = json.load(json_file)
        approximations = va_file['approximations']
        vector_ids = va_file['vector_ids']
    with open(input_folder + "\\partitions.pk", 'rb') as pickle_file:
        partitions = pickle.load(pickle_file)
    with open(input_folder + "\\vectors.pk", 'rb') as pickle_file:
        vectors = pickle.load(pickle_file)

    # normal knn without VA files
    knn_image_ids = get_n_closest_images(query_vector,vectors,k,vector_ids,p_in_lp)

    query_approximation = create_approximation([query_vector],partitions)[0]
    print("query approximation=",query_approximation)

    knn_image_ids_va = []
    num_images_considered = 0
    if algorithm == 'va_ssa':
        print(colored("va_ssa",'blue'))
        knn_image_ids_va, num_images_considered = va_ssa(approximations, vectors, query_approximation, query_vector,k,partitions,vector_ids,p_in_lp)
        deliverable = {'top_k_va_files':list(knn_image_ids_va),"top_k_knn":knn_image_ids}
        print(colored(deliverable,'blue'))

    elif algorithm == 'va_noa':
        print(colored("va_noa not currently implemented", 'red'))

    miss_rate, false_positive_rate = calculate_statistics(knn_image_ids,knn_image_ids_va, num_images_considered)
    statistics = {'miss_rate': miss_rate, "false_positive_rate": false_positive_rate}
    num_dim = len(query_vector)
    approximation_len = len(query_approximation)
    output_filename = "va_miss_fp"+"_"+str(num_dim)+"_"+str(approximation_len)+".json"
    save_to_json(OUTPUTS_PATH+output_filename,statistics)

    return knn_image_ids_va

def va_files_execution(train_features, test_features):

    b = int(input("Number of bits per dimension for VA"))
    t = int(input("Number of nearest neighbors required to query image:"))
    # knn = []
    # t = 10
    for b in range(b,b+1):
        image_vectors = []
        image_ids = []
        for image_id in train_features:
            image_vectors.append(train_features[image_id])
            image_ids.append(image_id)
        image_vectors = np.asarray(image_vectors)
        print(colored("num features per image=" + str(len(image_vectors[0])),'blue'))
        query_ids = []
        query_vector = []
        for query_id in test_features:
            query_ids.append(query_id)
            query_vector.append(test_features[query_id])
        query_ids = query_ids[0]
        query_vector = np.array(query_vector[0])

        create_and_save_va_file(image_vectors,b,image_ids,OUTPUTS_PATH)
        knn = va_search(OUTPUTS_PATH,query_vector,t,'va_ssa',p_in_lp=2)
    return knn

def main():
    b = 2
    filename = 'C:\\Users\\fateh\\MS ASU\\Courses\\CSE 515 MWDB\\Project\\Phase3\\experimentation\\Outputs\\lda_transformed_dataset_task3.pk'
    # vec = load_vectors(filename)
    # print(vec,np.array(vec).shape)
    #
    # filename = 'C:\\Users\\fateh\\MS ASU\\Courses\\CSE 515 MWDB\\Project\\Phase3\\experimentation\\Outputs\\lda_imageids_task3.pk'
    # image_ids = load_vectors(filename)
    # print(image_ids)
    vec = np.array([[1,3],[2,3],[4,10],[13,6],[18,1]])
    image_ids = ['0','1','2','3','4']
    create_and_save_va_file(vec,b,image_ids,"..\\Outputs_VA")

    # partitions = get_partitions(vec,b)
    #
    # approximations = create_approximation(vec,partitions)
    # print('approximations=',approximations)
    # print("size=",sys.getsizeof(approximations))
    # query = [0.2, 0.4, 0.4]
    query = [2,4]
    # query = [[0.20,0.33],[0.30,0.60],[0.6,0.4]]
    va_search("..\\Outputs_VA",query,k=4,algorithm='va_ssa')

    # query_approximation = create_approximation(query, partitions)[2]
    # print('query_approximation=',query_approximation)
    # li,ui = get_bounds(query[2],partitions,query_approximation,approximations[1],2)
    # print("li=",li, "ui=",ui)


if __name__ == "__main__":
    main()