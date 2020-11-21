import argparse
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from phase2 import task0a, task0b, task1, task3


def construct_adjacency_graph(matrix, k):
    matrix = matrix * (matrix >= np.sort(matrix, axis=1)[:, [-k]])
    return matrix


def construct_restart_vector(vector_size, seed_data_columns):
    vector = np.zeros((vector_size, 1))
    if len(seed_data_columns) !=0:
        val = 1 / len(seed_data_columns)
        for c in seed_data_columns:
            vector[c][0] = val
    else:
        val = 1 / vector_size
        vector = val * np.ones((vector_size, 1))
    return vector


def ppr(normalized_graph, restart_vector):
    c = 0.45
    max_iterations = 0
    new_steady_state_prob = restart_vector
    steady_state_prob = np.zeros(restart_vector.shape)
    while max_iterations < 200 and not np.equal(steady_state_prob, new_steady_state_prob).all():
        steady_state_prob = new_steady_state_prob
        new_steady_state_prob = ((1 - c) * np.dot(normalized_graph, steady_state_prob)) + (c * restart_vector)
        max_iterations += 1

    # a = np.array([1,2,3])
    # b = np.array([[1],[2],[3]])
    # print(np.dot(a,b))

    print("no of iterations to converge: ", max_iterations)
    return steady_state_prob


def visualize_dominant_features(gestures_dir, gestures):
    for file in gestures:
        for comp in ['X/', 'Y/', 'Z/', 'W/']:
            data = pd.read_csv(gestures_dir + comp + str(file).replace("_words", ""), header=None).transpose()
            data.plot.line()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')

    parser.add_argument('--gestures_dir', help='directory of input data', required=True)
    parser.add_argument('--k', type=int, help='k most similar gestures', required=True)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', default="pca", required=True)
    parser.add_argument('--user_option_k', type =int, help='Number of reduced dimensions', default=20, required=True)
    parser.add_argument('--n', nargs='+', help='User specified gestures or seed data', required=False)
    parser.add_argument('--m', type=int, help='Number of dominant features', required=True)

    # optional parameters
    parser.add_argument('--window', type=int, help='window length', required=False)
    parser.add_argument('--shift', type=int, help='shift length', required=False)
    parser.add_argument('--resolution', type=int, help='resolution', required=False)
    parser.add_argument('--output_dir', help='output directory', default="../outputs/", required=False)
    parser.add_argument('--vector_model', help='vector model', required=False)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', required=False)

    args = parser.parse_args()

    task0a.call_task0a(args.gestures_dir, args.window, args.shift, args.resolution)  # construct words from data
    task0b.call_task0b(args.output_dir)  # construct tf and tf-idf
    task1.call_task1(args.output_dir, args.vector_model, args.user_option, args.user_option_k)  # get for pca trained model
    task3.call_task3(args.vector_model, args.output_dir, args.user_option, 4,
                     "svd", args.custom_cost)  # construct gesture_gesture_similarity matrix

    # -------------------------------------- Personalized page rank algorithm  ----------------------------------------

    similarity_matrix = np.array(
        pd.read_csv(args.output_dir + "similarity_matrix_" + args.user_option + ".csv", header=None))

    adjacency_graph = construct_adjacency_graph(np.array(similarity_matrix[1:, 1:].tolist(), dtype=float),
                                                args.k)  # construct adjacency graph

    column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name

    name_column_map = dict()  # give a filename, returns the row index
    for index, filename in enumerate(column_file_map):
        name_column_map[filename] = index

    user_specified_columns = []
    if args.n:
        user_specified_columns = [name_column_map[x] for x in args.n]

    print("User specified features are ", args.n)

    restart_vector = construct_restart_vector(len(adjacency_graph), user_specified_columns)

    # sum of absolute values for each col equals to one TODO: decide on which norm to use here
    normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)

    steady_state_prob_vector = ppr(normalized_adjacency_graph, restart_vector)

    pd.DataFrame(steady_state_prob_vector).to_csv(args.output_dir + "steady_state_prob.csv", header=None, index=None)

    steady_state_prob_vector = steady_state_prob_vector.tolist()
    sorted_list = sorted(zip(steady_state_prob_vector, range(len(steady_state_prob_vector))), key=lambda v: v[0],
                         reverse=True)
    dominant_feature_indices = []
    for (s, i) in sorted_list:
        dominant_feature_indices.append(i)
    dominant_feature_indices = dominant_feature_indices[:args.m]
    print(dominant_feature_indices)

    dominant_features = [column_file_map[i] for i in dominant_feature_indices]
    print("Dominant features ", dominant_features)

    # -----------------------------------------------  visualize dominant features ------------------------------------
    # visualize_dominant_features(args.gestures_dir, dominant_features)
