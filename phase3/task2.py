from phase2 import task0a, task0b, task1, task3
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import sklearn
import csv
from phase3.task1 import ppr

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
    #labels_train = np.array(labels_train)
    labels_train = labels_train[:,1]
    labels_train = labels_train[:, np.newaxis]
    vectors = np.concatenate((filenames, labels_train, vectors_train[:,1:]), axis=1)

    for v in vectors_test :
        neighbours = get_n_nearest(v[1:], vectors, nn)
        #print(neighbours)
        mode = calc_mode(neighbours)
        predictions.append([v[0], mode])

    return predictions

def calc_accuracy(predicted, test) :
    fp = 0
    for i in range(len(test)) :
        if(predicted[i]==test[i]) :
            fp += 1
    accuracy = fp / len(test)
    print("Accuracy : ", accuracy)


def ppr_2(query_file, labels, vector_model, output_dir, user_option, custom_cost, k):

    task3.call_task3(vector_model, output_dir, user_option, 4,
                     "svd", custom_cost)  # construct gesture_gesture_similarity matrix
    similarity_matrix_df = pd.read_csv(output_dir + "similarity_matrix_" + user_option + ".csv", header=None, low_memory=False)

    similarity_matrix = np.array(similarity_matrix_df)
    column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name

    name_column_map = dict()  # give a filename, returns the row index
    for index, filename in enumerate(column_file_map):
        name_column_map[filename] = index

    adjacency_graph = np.array(similarity_matrix[1:, 1:].tolist(), dtype=float)
    adjacency_graph = adjacency_graph * (adjacency_graph >= np.sort(adjacency_graph, axis=1)[:, [-k]])
    normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)
    vector_size = len(adjacency_graph)

    class1 = ["1_words.csv", "2_words.csv", "3_words.csv", "4_words.csv", "5_words.csv", "6_words.csv", "7_words.csv",
              "8_words.csv", "9_words.csv", "10_words.csv","11_words.csv", "12_words.csv", "13_words.csv", "14_words.csv", "15_words.csv", "16_words.csv", "17_words.csv",
              "18_words.csv", "19_words.csv", "20_words.csv","21_words.csv", "22_words.csv", "23_words.csv", "24_words.csv", "25_words.csv", "26_words.csv", "27_words.csv",
              "28_words.csv", "29_words.csv", "30_words.csv","31_words.csv"]
    restart_vector_class1 = np.zeros((vector_size, 1))
    for f in class1:
        column = name_column_map[f]
        restart_vector_class1[column][0] = 1
    ppr_vector_class1 = ppr(normalized_adjacency_graph, restart_vector_class1)

    class2 = ["249_words.csv", "250_words.csv", "251_words.csv", "252_words.csv", "253_words.csv", "254_words.csv", "255_words.csv",
              "256_words.csv", "257_words.csv", "258_words.csv","259_words.csv", "260_words.csv", "261_words.csv", "262_words.csv", "263_words.csv", "264_words.csv", "265_words.csv",
              "266_words.csv", "267_words.csv", "268_words.csv","269_words.csv", "270_words.csv", "271_words.csv", "272_words.csv", "273_words.csv", "274_words.csv", "275_words.csv",
              "276_words.csv", "277_words.csv", "278_words.csv","279_words.csv"]
    restart_vector_class2 = np.zeros((vector_size, 1))
    for f in class2:
        column = name_column_map[f]
        restart_vector_class2[column][0] = 1
    ppr_vector_class2 = ppr(normalized_adjacency_graph, restart_vector_class2)

    class3 = ["559_words.csv", "560_words.csv", "561_words.csv","562_words.csv", "563_words.csv", "564_words.csv", "565_words.csv",
              "566_words.csv","567_words.csv", "568_words.csv", "569_words.csv","570_words.csv", "571_words.csv", "572_words.csv", "573_words.csv", "574_words.csv", "575_words.csv",
              "576_words.csv","577_words.csv", "578_words.csv", "579_words.csv","580_words.csv", "581_words.csv", "582_words.csv", "583_words.csv", "584_words.csv", "585_words.csv",
              "586_words.csv","587_words.csv", "588_words.csv", "589_words.csv"]
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
    print("Based on PPR classifier-2 for given query the class label", label_map[label])
    # --------------------------------------------- CALCULATING ACCURACY----------------------------------
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
        if (label == class_map[query_file]):
            count += 1
    accuracy_ppr_classifier_2 = count / len(name_column_map)
    print("Accuracy score: ", accuracy_ppr_classifier_2)

def ppr_classifier(labels, vector_model, output_dir, user_option, custom_cost, k):

    output_file = open(output_dir + "ppr_classification.txt", "w")

    number_of_dominant_features = 10

    task3.call_task3(vector_model, output_dir, user_option, 4,
                     "svd", custom_cost)  # construct gesture_gesture_similarity matrix

    similarity_matrix_df = pd.read_csv(output_dir + "similarity_matrix_" + user_option + ".csv", header=None,
                                       low_memory=False)

    similarity_matrix = np.array(similarity_matrix_df)
    column_file_map = similarity_matrix[0][1:].tolist()  # give a column number, return file name
    name_column_map = dict()  # give a filename, returns the row index
    for index, filename in enumerate(column_file_map):
        name_column_map[filename] = index

    # ---------------------------------------construct adjacency matrix ---------------------------------

    labelled_gestures = ["1_words.csv", "2_words.csv", "3_words.csv", "4_words.csv", "5_words.csv", "6_words.csv", "7_words.csv",
                         "8_words.csv", "9_words.csv", "10_words.csv", "11_words.csv", "12_words.csv", "13_words.csv",
                         "14_words.csv", "15_words.csv", "16_words.csv", "17_words.csv", "18_words.csv", "19_words.csv",
                         "20_words.csv", "21_words.csv", "22_words.csv", "23_words.csv","24_words.csv", "25_words.csv",
                         "26_words.csv", "27_words.csv","28_words.csv", "29_words.csv", "30_words.csv", "31_words.csv", "249_words.csv","250_words.csv",
                         "251_words.csv","252_words.csv","253_words.csv","254_words.csv","255_words.csv","256_words.csv","257_words.csv","258_words.csv",
                         "259_words.csv","260_words.csv","261_words.csv","262_words.csv","263_words.csv","264_words.csv","265_words.csv","266_words.csv",
                         "267_words.csv","268_words.csv","269_words.csv","270_words.csv","271_words.csv", "272_words.csv", "273_words.csv", "274_words.csv", "275_words.csv",
              "276_words.csv", "277_words.csv", "278_words.csv","279_words.csv","559_words.csv", "560_words.csv", "561_words.csv","562_words.csv", "563_words.csv", "564_words.csv", "565_words.csv",
              "566_words.csv","567_words.csv", "568_words.csv", "569_words.csv","570_words.csv", "571_words.csv", "572_words.csv", "573_words.csv", "574_words.csv", "575_words.csv",
              "576_words.csv","577_words.csv", "578_words.csv", "579_words.csv","580_words.csv", "581_words.csv", "582_words.csv", "583_words.csv", "584_words.csv", "585_words.csv",
              "586_words.csv","587_words.csv", "588_words.csv", "589_words.csv"]
    labelled_gestures_classes = ['vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene',
                                 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene',
                                 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'vattene',
                                 'vattene', 'vattene', 'vattene', 'vattene', 'vattene', 'combinato', 'combinato', 'combinato', 'combinato',
                                 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato',
                                 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato',
                                 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato', 'combinato',
                                 'combinato', 'combinato', 'combinato', 'combinato', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo',
                                 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo',
                                 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo',
                                 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo', 'daccordo']

    file_name_class_label_map = {}
    for j, file in enumerate(labelled_gestures):
        file_name_class_label_map[file] = labelled_gestures_classes[j]

    labelled_gesture_columns = [name_column_map[x]+1 for x in labelled_gestures]

    unlabelled_gestures = list(set(column_file_map) - set(labelled_gestures))
    unlabelled_gesture_columns = [name_column_map[x]+1 for x in unlabelled_gestures]

    csv_write = csv.writer(output_file)

    for c_index, c in enumerate(unlabelled_gesture_columns):

        matrix_columns = labelled_gesture_columns + [c]
        adjacency_graph = similarity_matrix_df.loc[matrix_columns, matrix_columns]
        adjacency_graph = adjacency_graph.astype(dtype=float)
        # TODO: Do we have to consider only k most closest here?
        # adjacency_graph = adjacency_graph * (adjacency_graph >= np.sort(adjacency_graph, axis=1)[:, [-k]])
        normalized_adjacency_graph = sklearn.preprocessing.normalize(adjacency_graph, norm='l1', axis=0)
        vector_size = len(adjacency_graph)
        restart_vector = np.zeros((vector_size, 1))
        restart_vector[vector_size-1][0] = 1
        ppr_vector = ppr(normalized_adjacency_graph, restart_vector)
        matrix_file_names = [column_file_map[k-1] for k in matrix_columns]
        dominant_file_names = sorted(zip(ppr_vector, matrix_file_names), key=lambda v: v[0], reverse=True)
        dominant_features_class = []
        for ranking, filename in dominant_file_names:
            if filename != column_file_map[c-1]:
                dominant_features_class.append(file_name_class_label_map[filename])

        class_label = max(set(dominant_features_class), key=dominant_features_class.count)
        csv_write.writerow((unlabelled_gestures[c_index], class_label))

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))        
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

def labels_str_int(labels) :
    classes = {}
    labels_int = []
    count = 0
    for i in range(labels.size) :
        if labels[i] not in classes :
            classes[labels[i]] = count
            count += 1
        labels_int.append(classes[labels[i]])
        
    return classes, np.array(labels_int)

def labels_int_str(labels, cmap) :
    labels_str = []
    for l in labels :
        for c in cmap :
            if cmap[c]==l :
                labels_str.append(str(c))
    return labels_str
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')
    parser.add_argument('--query_file', help='Query file name', default = '250.csv', required=False)

    parser.add_argument('--nn', type=int, help='number of neighbours', default = 30, required=False)

    parser.add_argument('--gestures_dir', help='directory of input data', default = '../Phase3_data_for_report/', required=False)
    parser.add_argument('--k', type=int, help='reduced vectors', default = 20, required=False)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', default = 'pca', required=False)

    # optional parameters
    parser.add_argument('--window', type=int, help='window length', default=3, required=False)
    parser.add_argument('--shift', type=int, help='shift length', default=3, required=False)
    parser.add_argument('--resolution', type=int, help='resolution', default=3, required=False)
    parser.add_argument('--output_dir', help='output directory', default='../outputs/', required=False)
    parser.add_argument('--vector_model', help='vector model', default='tf', required=False)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', default = False, required=False)

    args = parser.parse_args()

    task0a.call_task0a(args.gestures_dir, args.window, args.shift, args.resolution)
    task0b.call_task0b(args.output_dir)
    task1.call_task1(args.output_dir, args.vector_model, args.user_option, args.k)
    
    
    vectors = pd.read_csv(args.output_dir + args.vector_model + "_" + args.user_option + "_vectors.csv", header = None, low_memory=False)
    vectors = vectors.replace({0: r'(_words.csv)'}, { 0 : ""}, regex=True)
    vectors = np.array(vectors)
    filenames = vectors[:, 0]
  
    labels_raw = np.array(pd.read_csv(args.gestures_dir + 'all_labels.csv', index_col=None, header=None))
    labels_train = np.array(pd.read_csv(args.gestures_dir + 'training_labels.csv', index_col = None, header=None))
    
    vectors_train = []
    for l in labels_train :
        for v in vectors :
            n = str(l[0])
            if (n==v[0]) :
                vectors_train.append(v)
                break
    
    vectors_test = []
    for v in vectors :
        for vt in vectors_train :
            present = 0 
            if np.array_equal(v, vt) :
                present = 1
                break
        if not present :
            vectors_test.append(v)
    vectors_train = np.array(vectors_train)
    vectors_test = np.array(vectors_test)
    
    labels_test = []
    for v in vectors_test :
        for l in labels_raw :
            if(v[0]==l[0]) :
                labels_test.append(l)
                break
    labels_test = np.array(labels_test)
            
    labels_predicted = knn(vectors_train, vectors_test, labels_train, args.nn)
    labels_predicted = np.array(labels_predicted)
    print("K-nearest neighbours")
    calc_accuracy(labels_predicted[:,1], labels_test[:,1])
        
    
    cmap, labels_train_int = labels_str_int(labels_train[:,1])    
    decisiontree = DecisionTreeClassifier(max_depth=10)
    decisiontree.fit(vectors_train[:,1:], labels_train_int)
    
    #print(vectors_test[:,1:])
    labels_predicted = decisiontree.predict(vectors_test[:,1:])
    labels_predicted = labels_int_str(labels_predicted, cmap)
    labels_predicted = np.array(labels_predicted)
    print("decision tree")
    calc_accuracy(labels_predicted, labels_test[:,1])

    print("PPR CLASSIFICATION - I")
    ppr_classifier(np.array(labels_raw), args.vector_model, args.output_dir, args.user_option,
                   args.custom_cost, args.k)
    print("PPR CLASSIFICATION - II")
    ppr_2(args.query_file, np.array(labels_raw), args.vector_model, args.output_dir, args.user_option, args.custom_cost,
          args.k)
    