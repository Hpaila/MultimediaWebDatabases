import numpy as np
import pandas as pd
import argparse
from scipy.spatial import distance
from task3 import get_t_closest_gestures

def convert(val):
    if val > 0:
        return 1
    return 0

def get_updated_gestures(relevant_gestures, irrelevant_gestures, t, initial_search_results, vectors_path):
    #By default, using tf_idf vectors
    input_vectors_df = pd.read_csv(vectors_path)
    input_vectors_with_labels = np.array(input_vectors_df)
    input_vectors = input_vectors_with_labels[0:, 1:]

    terms = input_vectors_df.columns
    terms = terms[1:]
    
    N = len(input_vectors)
    d = input_vectors.shape[1]
    
    binary_weights_matrix = np.vectorize(convert)(input_vectors)

    binary_vectors_map = {}
    for i in range(len(input_vectors_with_labels)):
        binary_vectors_map[input_vectors_with_labels[i][0]] = binary_weights_matrix[i]

    R = len(relevant_gestures)
    N = len(irrelevant_gestures)

    r = np.array([0 for i in range(d)])
    for relevant_gesture in relevant_gestures:
        r += binary_vectors_map[relevant_gesture]
    
    # N = t
    # n = np.array([0 for i in range(d)])
    # for i in range(t):
    #     n += binary_vectors_map[initial_search_results[i]]

    n = np.array([0 for i in range(d)])
    for irrelevant_gesture in irrelevant_gestures:
        n += binary_vectors_map[irrelevant_gesture]

    p = np.divide(r + 0.5, R+1)
    u = np.divide(n + 0.5, N+1)
    log_values = np.log(np.divide(p*(1-u), u*(1-p)))

    sorted_boost_values, sorted_terms = zip(*sorted(zip(log_values, terms), reverse=True))
    with open("phase3/task4_output", "w") as f:
        for i in range(len(sorted_boost_values)):
            f.write(str(terms[i]) + " " + str(sorted_boost_values[i]))
            f.write("\n")

    similarity_map = {}
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
    parser.add_argument('--vectors', help='input vectors path', required=True)
    args = parser.parse_args()

    query_gesture = args.query_gesture + "_words.csv"
    
    #initial run
    initial_search_results = get_t_closest_gestures(6, 3, args.vectors, args.t, query_gesture)

    print("Initial search results are")
    for result in initial_search_results:
        print(result)

    previous_search_results = initial_search_results
    while True:
        relevant_gestures = input("Enter the relevant gestures if any\n")
        if "none" in relevant_gestures:
            relevant_gestures = []
        else:
            relevant_gestures = relevant_gestures.split(" ")
            relevant_gestures = [gesture + "_words.csv" for gesture in relevant_gestures]

        irrelevant_gestures = input("Enter the irrelevant gestures if any\n")
        if "none" in irrelevant_gestures:
            irrelevant_gestures = []
        else:
            irrelevant_gestures = irrelevant_gestures.split(" ")
            irrelevant_gestures = [gesture + "_words.csv" for gesture in irrelevant_gestures]

        updated_results = get_updated_gestures(relevant_gestures, irrelevant_gestures, args.t, previous_search_results, args.vectors)
        previous_search_results = updated_results
        print("Revised results are as below...")
        for res in updated_results:
            print(res)