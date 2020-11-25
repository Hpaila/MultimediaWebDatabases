# import sys
# sys.path.append("/Users/rahulsanjay/Downloads/MultimediaWebDatabases")

import argparse
import sklearn
import numpy as np
import pandas as pd
from phase2 import task1,task3
from phase3.task1 import ppr


def get_updated_gestures_task5(relevant_gestures, irrelevant_gestures, t, query_gesture):
    print(query_gesture)

    data_file_name = "outputs/tf_idf_pca_vectors.csv"
    similarity_matrix_file_name = "outputs/similarity_matrix_pca.csv"
    data_matrix = np.array(pd.read_csv(data_file_name, header=None))
    query_gesture_row_index = np.where(data_matrix == query_gesture)[0][0]
    graph_degree = 10
    task3.call_task3("tf_idf", "outputs/", "pca", 4, "svd", "False")
    relevant_gesture_row_indices = []

    irrelevant_gestures_vector = np.zeros((1, len(data_matrix[0]) - 1), dtype=object)
    relevant_gestures_vector = np.zeros((1, len(data_matrix[0]) - 1), dtype=object)
    query_gesture_values = data_matrix[query_gesture_row_index, 1:].astype(np.float)
    if relevant_gestures:
        for gesture in relevant_gestures:
                gesture_row_index = np.where(data_matrix == gesture)[0][0]
                relevant_gestures_vector = np.add(relevant_gestures_vector, data_matrix[gesture_row_index, 1:].astype(np.float))
                relevant_gesture_row_indices.append(gesture_row_index)
        relevant_gestures_vector = (1 / (len(relevant_gestures))) * relevant_gestures_vector

    if irrelevant_gestures:
        for gesture in irrelevant_gestures:
                # print(np.where(data_matrix == gesture)[0])
                gesture_row_index = np.where(data_matrix == gesture)[0][0]
                irrelevant_gestures_vector = np.add(irrelevant_gestures_vector, data_matrix[gesture_row_index, 1:].astype(np.float))
        irrelevant_gestures_vector = (-1 / len(irrelevant_gestures)) * irrelevant_gestures_vector

    data_matrix[query_gesture_row_index, 1:] = np.add(query_gesture_values, relevant_gestures_vector,
                                                      irrelevant_gestures_vector)
    pd.DataFrame(data_matrix).to_csv(data_file_name, header=None, index=None)

    similarity_matrix = np.array(
        pd.read_csv(similarity_matrix_file_name, header=None))

    column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name

    name_column_map = dict()  # give a filename, returns the row index
    for index, filename in enumerate(column_file_map):
        name_column_map[filename] = index

    adjacency_graph = np.array(similarity_matrix[1:, 1:].tolist(), dtype=float)
    adjacency_graph = adjacency_graph * (adjacency_graph >= np.sort(adjacency_graph, axis=1)[:, [-graph_degree]])
    normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)

    restart_vector = np.zeros((len(adjacency_graph), 1))
    restart_vector[query_gesture_row_index][0] = 1
    for i in relevant_gesture_row_indices:
        restart_vector[i][0] = 1
    ppr_vector = ppr(normalized_adjacency_graph, restart_vector)

    sorted_list = sorted(zip(ppr_vector, range(len(ppr_vector))), key=lambda v: v[0],
                         reverse=True)
    dominant_feature_indices = []
    for (s, i) in sorted_list:
        dominant_feature_indices.append(i)
    dominant_feature_indices = dominant_feature_indices[:t]

    dominant_features = [column_file_map[i] for i in dominant_feature_indices]
    # print("Dominant features ", dominant_features)
    return dominant_features

def initial_result_task5 (vectors, t, query_gesture):
    data_file_name = vectors
    task1.call_task1("outputs/", "tf_idf", "pca", 10)
    task3.call_task3("tf_idf", "outputs/", "pca", 4, "svd", "False")
    similarity_matrix_file_name = "outputs/similarity_matrix_pca.csv"
    data_matrix = np.array(pd.read_csv(data_file_name, header=None))
    query_gesture_row_index = np.where(data_matrix == query_gesture)[0][0]
    graph_degree = 10

    similarity_matrix = np.array(
        pd.read_csv(similarity_matrix_file_name, header=None))

    column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name

    name_column_map = dict()  # give a filename, returns the row index
    for index, filename in enumerate(column_file_map):
        name_column_map[filename] = index

    adjacency_graph = np.array(similarity_matrix[1:, 1:].tolist(), dtype=float)
    adjacency_graph = adjacency_graph * (adjacency_graph >= np.sort(adjacency_graph, axis=1)[:, [-graph_degree]])
    normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)

    restart_vector = np.zeros((len(adjacency_graph), 1))
    restart_vector[query_gesture_row_index][0] = 1
    ppr_vector = ppr(normalized_adjacency_graph, restart_vector)
    sorted_list = sorted(zip(ppr_vector, range(len(ppr_vector))), key=lambda v: v[0],
                         reverse=True)
    dominant_feature_indices = []
    for (s, i) in sorted_list:
        dominant_feature_indices.append(i)
    dominant_feature_indices = dominant_feature_indices[:t]
    # print("indices",dominant_feature_indices)
    dominant_features = [column_file_map[i] for i in dominant_feature_indices]
    print("Dominant features ", dominant_features)
    return dominant_features



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PPR based relevance feedback')
    parser.add_argument('--query_gesture', help='query gesture', required=True)
    parser.add_argument('--t', type=int, help='get t most similar gestures', required=True)
    parser.add_argument('--vector_model', help='vector model', default='tf_idf', required=False)
    parser.add_argument('--gestures_dir', help='directory of input data', default='sample/', required=False)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', default='pca', required=False)
    parser.add_argument('--output_dir', help='output directory', default='outputs/', required=False)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', required=False)
    parser.add_argument('--user_option_k', type=int, help='Number of reduced dimensions', default=10, required=False)

    args = parser.parse_args()

    task1.call_task1(args.output_dir, args.vector_model, args.user_option, args.user_option_k)
    task3.call_task3(args.vector_model, args.output_dir, args.user_option, 4,
                     "svd", args.custom_cost)
    query_gesture = args.query_gesture + "_words.csv"
    data_file_name = args.output_dir+args.vector_model+"_"+args.user_option+"_vectors.csv"
    similarity_matrix_file_name = args.output_dir + "similarity_matrix_" + args.user_option + ".csv"
    data_matrix = np.array(pd.read_csv(data_file_name, header=None))
    query_gesture_row_index = np.where(data_matrix == query_gesture)[0][0]
    # print(query_gesture_row_index, " is row index")
    graph_degree = 10

    relevant_gesture_row_indices = []

    while True:

        query_gesture_values = data_matrix[query_gesture_row_index, 1:]

        similarity_matrix = np.array(
            pd.read_csv(similarity_matrix_file_name, header=None))

        column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name

        name_column_map = dict()  # give a filename, returns the row index
        for index, filename in enumerate(column_file_map):
            name_column_map[filename] = index

        adjacency_graph = np.array(similarity_matrix[1:, 1:].tolist(), dtype=float)
        adjacency_graph = adjacency_graph * (adjacency_graph >= np.sort(adjacency_graph, axis=1)[:, [-graph_degree]])
        normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)

        restart_vector = np.zeros((len(adjacency_graph), 1))
        restart_vector[query_gesture_row_index][0] = 1
        for i in relevant_gesture_row_indices:
            # print(i)
            restart_vector[i][0] = 1
        relevant_gesture_row_indices = []

        ppr_vector = ppr(normalized_adjacency_graph, restart_vector)

        sorted_list = sorted(zip(ppr_vector, range(len(ppr_vector))), key=lambda v: v[0],
                             reverse=True)
        dominant_feature_indices = []
        for (s, i) in sorted_list:
            dominant_feature_indices.append(i)
        dominant_feature_indices = dominant_feature_indices[:args.t]

        dominant_features = [column_file_map[i].replace("_words.csv", "") for i in dominant_feature_indices]
        print("Dominant features ", dominant_features)

        irrelevant_gestures_vector = np.zeros((1, len(data_matrix[0]) - 1), dtype=object)
        relevant_gestures_vector = np.zeros((1, len(data_matrix[0]) - 1), dtype=object)

        relevant_gestures = input("Enter the relevant gestures if any\n")
        if relevant_gestures:
            relevant_gestures = relevant_gestures.split(" ")
            relevant_gestures = [gesture + "_words.csv" for gesture in relevant_gestures]
            for gesture in relevant_gestures:
                gesture_row_index = np.where(data_matrix == gesture)[0][0]
                # print("adding ", gesture_row_index)
                relevant_gestures_vector = np.add(relevant_gestures_vector, data_matrix[gesture_row_index, 1:])
                relevant_gesture_row_indices.append(gesture_row_index)
            relevant_gestures_vector = (1/(len(relevant_gestures))) * relevant_gestures_vector

        irrelevant_gestures = input("Enter the irrelevant gestures if any\n")
        if irrelevant_gestures:
            irrelevant_gestures = irrelevant_gestures.split(" ")
            irrelevant_gestures = [gesture + "_words.csv" for gesture in irrelevant_gestures]
            for gesture in irrelevant_gestures:
                # print(np.where(data_matrix == gesture)[0])
                gesture_row_index = np.where(data_matrix == gesture)[0][0]
                irrelevant_gestures_vector = np.add(irrelevant_gestures_vector, data_matrix[gesture_row_index, 1:])
            irrelevant_gestures_vector = (-1/len(irrelevant_gestures)) * irrelevant_gestures_vector

        data_matrix[query_gesture_row_index, 1:] = np.add(query_gesture_values, relevant_gestures_vector, irrelevant_gestures_vector)
        pd.DataFrame(data_matrix).to_csv(data_file_name, header=None, index=None)


