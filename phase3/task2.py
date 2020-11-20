from phase2 import task0a, task0b, task1
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')
    parser.add_argument('--query_file', help='Query file name', default = '2.csv', required=False)
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
    max_val = max(classes.values())
    max_key = [k for k, v in classes.items() if v == max_val]    
    
    print('Query file ', args.query_file, ' is ', max_key[0])    
    
    
    
    
        
    
        
    