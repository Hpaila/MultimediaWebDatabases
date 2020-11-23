from phase2 import task0a, task0b, task1, task3
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
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


def get_n_nearest(query_vector, vectors, nn):
    distances = {}
    for v in vectors:
        distances[v[0]] = [v[1], distance.euclidean(query_vector, v[2:])]
    sort_orders = sorted(distances.items(), key=lambda x: x[1][1])
    # for i in range(0,nn):
    #    print(sort_orders[i])
    return sort_orders[:nn]

def calc_mode(data) :
    classes = {}
    for d in data :
        if d[1][0] in classes :
            classes[d[1][0]] += 1
        else :
            classes[d[1][0]] = 1
    max_val = max(classes.values())
    max_keys = [k for k, v in classes.items() if v == max_val]
    return max_keys[0]


def knn(vectors_train, vectors_test, labels_train, nn) :

    predictions = []
    filenames = vectors_train[:,0]
    filenames = filenames[:, np.newaxis]
    labels_train = np.array(labels_train)
    labels_train = labels_train[:, np.newaxis]
    vectors = np.concatenate((filenames, labels_train, vectors_train[:,1:]), axis=1)

    for v in vectors_test :
        neighbours = get_n_nearest(v[1:], vectors, nn)
        mode = calc_mode(neighbours)
        predictions.append([v[0], mode])

    return predictions

def calc_accuracy(predicted, test) :
    fp = 0
    for i in range(len(test)) :
        if(predicted[i][1]==test[i]) :
            fp += 1
    accuracy = fp / len(test)
    print("Accuracy : ", accuracy)


def ppr_2(query_file, labels, vector_model, output_dir, user_option, custom_cost, k):
    print("classification 2:")
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

    class1 = ["1_words.csv", "2_words.csv", "3_words.csv", "4_words.csv", "5_words.csv", "6_words.csv", "7_words.csv",
              "8_words.csv", "9_words.csv", "10_words.csv"]
    restart_vector_class1 = np.zeros((vector_size, 1))
    for f in class1:
        column = name_column_map[f]
        restart_vector_class1[column][0] = 1
        ppr_vector_class1 = ppr(normalized_adjacency_graph, restart_vector_class1)

    class2 = ["249_words.csv", "250_words.csv", "251_words.csv", "252_words.csv", "253_words.csv", "254_words.csv",
              "255_words.csv",
              "256_words.csv", "257_words.csv", "258_words.csv"]
    restart_vector_class2 = np.zeros((vector_size, 1))
    for f in class2:
        column = name_column_map[f]
        restart_vector_class2[column][0] = 1
        ppr_vector_class2 = ppr(normalized_adjacency_graph, restart_vector_class2)

    class3 = ["580_words.csv", "581_words.csv", "582_words.csv", "583_words.csv", "584_words.csv", "585_words.csv",
              "586_words.csv",
              "587_words.csv", "588_words.csv", "589_words.csv"]
    restart_vector_class3 = np.zeros((vector_size, 1))
    for f in class3:
        column = name_column_map[f]
        restart_vector_class3[column][0] = 1
        ppr_vector_class3 = ppr(normalized_adjacency_graph, restart_vector_class3)

    query_file = query_file.replace(".csv", "_words.csv")
    user_specified_column = name_column_map[query_file]
    scores = [ppr_vector_class1[user_specified_column][0], ppr_vector_class2[user_specified_column][0],
              ppr_vector_class3[user_specified_column][0]]
    label_map = {0: "vattene", 1: "combinato", 2: "daccordo"}
    label = scores.index(max(scores))
    # print(scores)
    print("Classification_2 result:", label_map[label])
    print("accuracy:")
    labels.tolist()
    class_map = {}
    for label in labels:
        class_map[label[0]] = label[1]
    count = 0
    for f in name_column_map.keys():
        # query_file = f.replace(".csv", "_words.csv")
        column = name_column_map[f]
        scores = [ppr_vector_class1[column][0], ppr_vector_class2[column][0],
                  ppr_vector_class3[column][0]]
        label_map = {0: "vattene", 1: "combinato", 2: "daccordo"}
        label = label_map[scores.index(max(scores))]
        query_file = f.replace("_words.csv", "")
        if (label == class_map[int(query_file)]):
            count += 1
    print(count / len(name_column_map))


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
    # print(ppr_vector[0])
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

    print("Based on PPR classifier for given query the class label is ",
          max(set(dominant_features_class), key=dominant_features_class.count))

    print("accuracy:")
    count = 0
    for f in name_column_map.keys():
        # query_file = f.replace(".csv", "_words.csv")
        column = name_column_map[f]
        restart_vector = np.zeros((vector_size, 1))
        restart_vector[column][0] = 1
        ppr_vector = ppr(normalized_adjacency_graph, restart_vector)
        sorted_list = sorted(zip(ppr_vector, range(len(ppr_vector))), key=lambda v: v[0],
                             reverse=True)
        dominant_feature_indices = []
        for (s, i) in sorted_list:
            dominant_feature_indices.append(i)
        dominant_feature_indices = dominant_feature_indices[:number_of_dominant_features]

        dominant_features = [column_file_map[i].replace("_words.csv", "") for i in dominant_feature_indices]
        dominant_features_class = [class_map[int(x)] for x in dominant_features]
        label = max(set(dominant_features_class), key=dominant_features_class.count)
        query_file = f.replace("_words.csv", "")
        if (label == class_map[int(query_file)]):
            count += 1
    print(count / len(name_column_map))


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

    labels_raw = np.array(pd.read_csv(args.gestures_dir + 'all_labels.csv', index_col=None, header=None))
    labels_dict = {l[0] : l[1] for l in labels_raw}
    labels_ordered = [labels_dict[int(v[0].split('_')[0])] for v in vectors]

    vectors_train, vectors_test, labels_train, labels_test = train_test_split(vectors, labels_ordered, test_size=0.33, random_state=42)
    labels_predicted = knn(vectors_train, vectors_test, labels_train, args.nn)
    print(labels_predicted)
    calc_accuracy(labels_predicted, labels_test)

    ppr_classifier(args.query_file, np.array(labels_raw), args.vector_model, args.output_dir, args.user_option,
                   args.custom_cost, args.k)
    ppr_2(args.query_file, np.array(labels_raw), args.vector_model, args.output_dir, args.user_option, args.custom_cost,
          args.k)