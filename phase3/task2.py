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
def get_n_nearest(query_vector, vectors, nn) :
    distances={}
    for v in vectors:
        distances[v[0]]=[v[1], distance.euclidean(query_vector,v[2:])]
    sort_orders = sorted(distances.items(), key=lambda x: x[1][1])
    #for i in range(0,nn):
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
    parser.add_argument('--nn', type=int, help='number of neighbours', default = 8, required=False)
    
    parser.add_argument('--gestures_dir', help='directory of input data', default = '../Phase3_data_for_report/', required=False)
    parser.add_argument('--k', type=int, help='reduced vectors', default = 10, required=False)
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
    
    '''
    labels_raw = np.array(pd.read_csv(args.gestures_dir + 'all_labels.csv', index_col=None, header=None))
    labels_dict = {l[0] : l[1] for l in labels_raw}
    labels_ordered = [labels_dict[int(v[0].split('_')[0])] for v in vectors]
    '''
    
    labels_train = np.array(pd.read_csv('../sample_training_labels.csv', index_col = None, header=None))
    vectors_train = []
    for l in labels_train :
        for v in vectors :
            n = v[0].split('_')
            if ((l[0]== int(n[0])) and (n[1][0] == 'w')) :
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
    
    #vectors_train, vectors_test, labels_train, labels_test = train_test_split(vectors, labels_ordered, test_size=0.33, random_state=42)
    #labels_train = np.array(labels_train)
    
    labels_predicted = knn(vectors_train, vectors_test, labels_train, args.nn)
    
    '''
    print(labels_predicted)
    calc_accuracy(labels_predicted, labels_test)

    ppr_classifier(args.query_file, np.array(labels_raw), args.vector_model, args.output_dir, args.user_option, args.custom_cost, args.k)
    
    cmap, labels_train_int = labels_str_int(labels_train)    
    decisiontree = DecisionTreeClassifier(max_depth = 10)
    decisiontree.fit(vectors_train[:,1:], labels_train_int)
    
    #print(vectors_test[:,1:])
    labels_predicted = decisiontree.predict(vectors_test[:,1:])
    labels_predicted = labels_int_str(labels_predicted, cmap)
    calc_accuracy(labels_predicted, labels_test)

    print(labels_predicted)
    '''
    