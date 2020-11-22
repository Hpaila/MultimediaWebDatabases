from phase2 import task0a, task0b, task1
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
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
    
    
    
    
        
    
        
    