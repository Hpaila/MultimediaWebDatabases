from phase2 import task0a, task0b, task1, task3
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance
import sklearn

from phase3.task1 import ppr

'''
gestures_dir = '../sample/'
k = 20
user_option = 'pca'

window = 3
shift =3
resolution = 3
output_dir = '../outputs/'
vector_model = 'tf_idf'
custom_cost = False
'''
def get_n_nearest(query_vector, vectors, nn) :
    distances={}
    for v in vectors:
        distances[v[0]]=distance.euclidean(query_vector,v[1:])
    sort_orders = sorted(distances.items(), key=lambda x: x[1])
    for i in range(0,nn):
        print(sort_orders[i])
    return sort_orders[:10]


def ppr_classifier(query_file, labels, vector_model, output_dir, user_option, custom_cost, k):
    number_of_dominant_features = 10

    task3.call_task3(vector_model, output_dir, user_option, 4,
                     "svd", custom_cost)  # construct gesture_gesture_similarity matrix
    similarity_matrix = np.array(
        pd.read_csv(output_dir + "similarity_matrix_" + user_option + ".csv", header=None))

    column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name

    name_column_map = dict()  # give a filename, returns the row index
    for index, filename in enumerate(column_file_map):
        name_column_map[filename] = index

    adjacency_graph = np.array(similarity_matrix[1:, 1:].tolist(), dtype=float)
    adjacency_graph = adjacency_graph * (adjacency_graph >= np.sort(adjacency_graph, axis=1)[:, [-k]])
    normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)
    vector_size = len(adjacency_graph)
    restart_vector = np.zeros((vector_size, 1))

    query_file = query_file.replace(".csv", "_words.csv")
    user_specified_column = name_column_map[query_file]

    restart_vector[user_specified_column][0] = 1

    ppr_vector = ppr(normalized_adjacency_graph, restart_vector)

    sorted_list = sorted(zip(ppr_vector, range(len(ppr_vector))), key=lambda v: v[0],
                         reverse=True)
    dominant_feature_indices = []
    for (s, i) in sorted_list:
        dominant_feature_indices.append(i)
    dominant_feature_indices = dominant_feature_indices[:number_of_dominant_features]

    dominant_features = [column_file_map[i].replace("_words.csv", "") for i in dominant_feature_indices]
    print("Dominant features ", dominant_features)

    labels.tolist()
    class_map = {}
    for label in labels:
        class_map[label[0]] = label[1]

    dominant_features_class = [class_map[int(x)] for x in dominant_features]

    print("Based on PPR classifier for given query the class label is ", max(set(dominant_features_class), key=dominant_features_class.count))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')
    parser.add_argument('--query_file', help='Query file name', default = '250.csv', required=False)
    parser.add_argument('--nn', type=int, help='number of neighbours', default = 10, required=False)
    
    parser.add_argument('--gestures_dir', help='directory of input data', default = '../sample/', required=False)
    parser.add_argument('--k', type=int, help='reduced vectors', default = 20, required=False)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', default = 'pca', required=False)

    # optional parameters
    parser.add_argument('--window', type=int, help='window length', default=3, required=False)
    parser.add_argument('--shift', type=int, help='shift length', default=3, required=False)
    parser.add_argument('--resolution', type=int, help='resolution', default=3, required=False)
    parser.add_argument('--output_dir', help='output directory', default='../outputs/', required=False)
    parser.add_argument('--vector_model', help='vector model', default='tf_idf', required=False)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', default = False, required=False)
    
    args = parser.parse_args()
        
    task0a.call_task0a(args.gestures_dir, args.window, args.shift, args.resolution)
    task0b.call_task0b(args.output_dir)
    task1.call_task1(args.output_dir, args.vector_model, args.user_option, args.k)
    
    vectors = np.array(pd.read_csv(args.output_dir + args.vector_model + "_" + args.user_option + "_vectors.csv", header = None))
    filenames = vectors[:, 0]
    query_file_name = args.query_file.split('.')[0] + '_words.csv'
    query_file_index = np.where(filenames == query_file_name)
    query_vector = vectors[query_file_index[0][0], 1:]
    neighbours = get_n_nearest(query_vector, vectors, args.nn)
    
    labels = np.array(pd.read_csv(args.gestures_dir + 'all_labels.csv', index_col=None, header=None))
    classes = {}
    for n in neighbours :
        file = int(n[0].split('_')[0])
        i = np.where(labels == file)[0][0]
        if labels[i][1] not in classes :
            classes[labels[i][1]] = 1
        else :
            classes[labels[i][1]] += 1
    print("classes ", classes)
    max_val = max(classes.values())
    max_key = [k for k, v in classes.items() if v == max_val]    
    
    print('Based on K-NN classifier for given query file ', args.query_file, ' the assigned class is ', max_key[0])

    ppr_classifier(args.query_file, labels, args.vector_model, args.output_dir, args.user_option, args.custom_cost, args.k)
