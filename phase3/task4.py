import numpy as np
import pandas as pd
import argparse
from scipy.spatial import distance
from task3 import get_t_closest_gestures

def convert(val):
    if val > 0:
        return 1
    return 0

def get_updated_gestures(relevant_gestures, t, initial_search_results):
    #By default, using tf_idf vectors
    input_vectors_df = pd.read_csv("outputs/vectors/tf_idf_vectors.csv")
    columns = input_vectors_df.columns[1:]
    columns = [eval(column) for column in columns]

    input_vectors_with_labels = np.array(input_vectors_df)
    input_vectors = input_vectors_with_labels[0:, 1:]
    N = len(input_vectors)
    d = input_vectors.shape[1]
    
    binary_weights_matrix = np.vectorize(convert)(input_vectors)
    
    input_vectors_map = {}
    for i in range(len(input_vectors_with_labels)):
        input_vectors_map[input_vectors_with_labels[i][0]] = input_vectors[i]

    binary_vectors_map = {}
    for i in range(len(input_vectors_with_labels)):
        binary_vectors_map[input_vectors_with_labels[i][0]] = binary_weights_matrix[i]

    similarity_map = {}
    N = t
    R = len(relevant_gestures)
    
    r = np.array([0 for i in range(d)])
    for relevant_gesture in relevant_gestures:
        r += binary_vectors_map[relevant_gesture]
    
    n = np.array([0 for i in range(d)])
    for i in range(t):
        n += binary_vectors_map[initial_search_results[i]]
    
    factor = n/N
    p = np.divide(r + 0.5, R+1)
    u = np.divide(n - r + 0.5, N-R+1)
    log_values = np.log(np.divide(p*(1-u), u*(1-p)))

    for i in range(len(input_vectors)):
        similarity_map[input_vectors_with_labels[i][0]] = np.sum(binary_weights_matrix[i] * log_values)

    sort_orders = sorted(similarity_map.items(), key=lambda x: x[1], reverse=True)
    search_results = []
    for i in range(0,t):
        search_results.append(sort_orders[i][0])
    return search_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic relevance feedback')
    parser.add_argument('--query_gesture', help='query gesture', required=True)
    parser.add_argument('--t', type=int, help='get t most similar gestures', required=True)
    args = parser.parse_args()

    query_gesture = args.query_gesture + "_words.csv"
    
    #initial run
    initial_search_results = get_t_closest_gestures(6, 3, "outputs/vectors/tf_idf_vectors.csv", args.t, query_gesture)

    print("Initial search results are")
    for result in initial_search_results:
        print(result)

    #By default, using tf_idf vectors
    input_vectors_df = pd.read_csv("outputs/vectors/tf_idf_vectors.csv")
    columns = input_vectors_df.columns[1:]
    columns = [eval(column) for column in columns]

    input_vectors_with_labels = np.array(input_vectors_df)
    input_vectors = input_vectors_with_labels[0:, 1:]
    N = len(input_vectors)
    d = input_vectors.shape[1]
    
    binary_weights_matrix = np.vectorize(convert)(input_vectors)
    
    input_vectors_map = {}
    for i in range(len(input_vectors_with_labels)):
        input_vectors_map[input_vectors_with_labels[i][0]] = input_vectors[i]

    binary_vectors_map = {}
    for i in range(len(input_vectors_with_labels)):
        binary_vectors_map[input_vectors_with_labels[i][0]] = binary_weights_matrix[i]

    similarity_map = {}
    N = args.t
    while True:
        relevant_gestures = input("Enter the relevant gestures if any\n")
        if "none" in relevant_gestures:
            relevant_gestures = []
        else:
            relevant_gestures = relevant_gestures.split(" ")
            relevant_gestures = [gesture + "_words.csv" for gesture in relevant_gestures]

        R = len(relevant_gestures)
    
        r = np.array([0 for i in range(d)])
        for relevant_gesture in relevant_gestures:
            r += binary_vectors_map[relevant_gesture]
        
        n = np.array([0 for i in range(d)])
        for i in range(args.t):
            n += binary_vectors_map[initial_search_results[i]]
        
        factor = n/N
        p = np.divide(r + 0.5, R+1)
        u = np.divide(n - r + 0.5, N-R+1)
        log_values = np.log(np.divide(p*(1-u), u*(1-p)))

        for i in range(len(input_vectors)):
            similarity_map[input_vectors_with_labels[i][0]] = np.sum(binary_weights_matrix[i] * log_values)

        sort_orders = sorted(similarity_map.items(), key=lambda x: x[1], reverse=True)
        print("Modified results are")
        initial_search_results = []
        for i in range(0,args.t):
            initial_search_results.append(sort_orders[i][0])
            print(sort_orders[i][0])