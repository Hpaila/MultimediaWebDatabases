import numpy as np
import pandas as pd
import argparse
from scipy.spatial import distance

w = 0.3
def get_all_binary_strings(count):
    if count == 0:
        return ['', '']
    old = ['']
    new = []
    for i in range(count):
        for ele in old:
            new.append(ele + '0')
            new.append(ele + '1')
        old = list(new)
        new = []
    return old

def get_concatenated_hash(input_vector, p_stable_vectors, b):
    hash_values = (np.dot(input_vector, p_stable_vectors.T) > 0).astype("int")
    return ''.join(hash_values.astype("str"))

    # hash_values = np.floor((np.dot(input_vector, p_stable_vectors.T) + b) / w)
    # return ''.join(hash_values.astype("str"))

def get_t_closest_gestures(k, l, vectors, t, query_gesture):
    input_vectors_with_labels = None
    if "pca" in vectors or "svd" in vectors or "nmf" in vectors or "lda" in vectors:
        input_vectors_with_labels = np.array(pd.read_csv(vectors, header = None))
    else:
        input_vectors_with_labels = np.array(pd.read_csv(vectors))
    
    input_vectors_map = {}
    for input_vector in input_vectors_with_labels:
        input_vectors_map[input_vector[0]] = input_vector[1:]

    # Dimensionality of input vectors
    d = input_vectors_with_labels.shape[1] - 1
    p_stable_vectors_map = {}
    b_map = {}
    lsh_hash_tables = []

    for l in range(l):
        p_stable_vectors = np.random.randn(k, d)
        p_stable_vectors_map[l] = p_stable_vectors
        b = [np.random.uniform(0, w) for i in range(k)]
        b_map[l] = b
        for gesture_name, input_vector in input_vectors_map.items():
            concatenated_hash = get_concatenated_hash(input_vector, p_stable_vectors, b)
            lsh_hash_tables.append({})
            if concatenated_hash not in lsh_hash_tables[l]:
                lsh_hash_tables[l][concatenated_hash] = []
            lsh_hash_tables[l][concatenated_hash].append(gesture_name)

    number_of_buckets = 0
    distances = {}
    unique_gestures = set()
    overall_gestures = 0

    sort_orders = sorted(distances.items(), key=lambda x: x[1]) 
    count = 0
    while len(sort_orders) < t:
        number_of_buckets = 0
        distances = {}
        unique_gestures = set()
        overall_gestures = 0
        for i in range(l):
            query_concatened_hash = get_concatenated_hash(input_vectors_map[query_gesture], p_stable_vectors_map[i], b_map[i])
            sub_hash = query_concatened_hash[:-count]
            hashes = [sub_hash + binary_string for binary_string in get_all_binary_strings(count)]
            if count == 0:
                hashes = [query_concatened_hash]
            for query_hash in hashes:
                if query_hash in lsh_hash_tables[i]:
                    number_of_buckets += 1
                    gestures_in_bucket = lsh_hash_tables[i][query_hash]
                    for gesture in gestures_in_bucket:
                        overall_gestures += 1
                        unique_gestures.add(gesture)

        for gesture in unique_gestures:
            distances[gesture] = distance.euclidean(input_vectors_map[query_gesture], input_vectors_map[gesture])
            
        sort_orders = sorted(distances.items(), key=lambda x: x[1])
        if len(sort_orders) < t:
            # print("Found only ", len(sort_orders), " gestures in the buckets")
            # print("Reducing the value of k and searching again")
            count += 1

    res = []
    for i in range(0,t):
        res.append(sort_orders[i][0])
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Locality Sensitive Hashing')

    parser.add_argument('--k', type=int, help='Number of hashes per layer', required=True)
    parser.add_argument('--l', type=int, help='Number of layers', required=True)
    parser.add_argument('--vectors', help='input vectors path', required=True)

    args = parser.parse_args()
    print("Creating the LSH index structure...")

    input_vectors_with_labels = None
    if "pca" in args.vectors or "svd" in args.vectors or "nmf" in args.vectors or "lda" in args.vectors:
        input_vectors_with_labels = np.array(pd.read_csv(args.vectors, header = None))
    else:
        input_vectors_with_labels = np.array(pd.read_csv(args.vectors))
    
    input_vectors_map = {}
    for input_vector in input_vectors_with_labels:
        input_vectors_map[input_vector[0]] = input_vector[1:]

    # Dimensionality of input vectors
    d = input_vectors_with_labels.shape[1] - 1
    p_stable_vectors_map = {}
    b_map = {}
    lsh_hash_tables = []

    for l in range(args.l):
        p_stable_vectors = np.random.randn(args.k, d)
        p_stable_vectors_map[l] = p_stable_vectors
        b = [np.random.uniform(0, w) for i in range(args.k)]
        b_map[l] = b
        for gesture_name, input_vector in input_vectors_map.items():
            concatenated_hash = get_concatenated_hash(input_vector, p_stable_vectors, b)
            lsh_hash_tables.append({})
            if concatenated_hash not in lsh_hash_tables[l]:
                lsh_hash_tables[l][concatenated_hash] = []
            lsh_hash_tables[l][concatenated_hash].append(gesture_name)
        
    print("Created the LSH index structure")
    
    while True:
        query_gesture = input("Enter a query gesture to search\n")
        query_gesture += "_words.csv"
        t = int(input("Enter the value for t to get the t most similar gestures\n"))

        number_of_buckets = 0
        distances = {}
        unique_gestures = set()
        overall_gestures = 0

        sort_orders = sorted(distances.items(), key=lambda x: x[1]) 
        count = 0
        while len(sort_orders) < t:
            number_of_buckets = 0
            distances = {}
            unique_gestures = set()
            overall_gestures = 0
            for i in range(args.l):
                query_concatened_hash = get_concatenated_hash(input_vectors_map[query_gesture], p_stable_vectors_map[i], b_map[i])
                sub_hash = query_concatened_hash[:-count]
                hashes = [sub_hash + binary_string for binary_string in get_all_binary_strings(count)]
                if count == 0:
                    hashes = [query_concatened_hash]
                for query_hash in hashes:
                    if query_hash in lsh_hash_tables[i]:
                        number_of_buckets += 1
                        gestures_in_bucket = lsh_hash_tables[i][query_hash]
                        for gesture in gestures_in_bucket:
                            overall_gestures += 1
                            unique_gestures.add(gesture)

            for gesture in unique_gestures:
                distances[gesture] = distance.euclidean(input_vectors_map[query_gesture], input_vectors_map[gesture])
                
            sort_orders = sorted(distances.items(), key=lambda x: x[1])
            if len(sort_orders) < t:
                print("Found only ", len(sort_orders), " gestures in the buckets")
                print("Reducing the value of k and searching again")
                count += 1

        print("t most similar gestures are as below:")
        for i in range(0,t):
            print(sort_orders[i][0])

        print()
        print("Total number of buckets searched is ", number_of_buckets)
        print("Unique gestures considered ", len(unique_gestures))
        print("Overall gestures considered ", overall_gestures)






