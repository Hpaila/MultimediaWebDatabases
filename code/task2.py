import argparse
import os
import numpy as np
import pandas as pd
import csv
import glob
import joblib
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation, NMF
from scipy.spatial import distance
from sequence_utils import get_edit_distance, get_dtw_distance
import time

NUM_SENSORS = 20
#This is the total number of unique features in our dataset, so our vector size will be this for every database object or query object
feature_dict = set()

#This is a list of dictionaries, each dictionary contains words of one gesture and their respective counts
words_in_object_map = {}

#This is the total number of words in a given sensor (used in the denominator when calculating TF values)
total_count_of_words_in_object = {}

#Getting the number of files in the given directory of gestures (this gives us the number of objects in our database)
def get_number_of_gesture_files_in_dir(directory_path):
    return len(glob.glob1(directory_path,"*.csv"))

def process_word_file(directory_path, file_name):
    file_path = directory_path + file_name
    words = np.array(pd.read_csv(file_path, header = None))
    total_count_of_words_in_object[file_name] = len(words)
    words_in_object_map[file_name] = {}

    for row in words:
        sensor_id = row[1]
        word = [row[0], row[2]]
        word.extend(row[6:])
        word = tuple(word)
        feature_dict.add(word)
        if word not in words_in_object_map[file_name]:
            words_in_object_map[file_name][word] = 0
        words_in_object_map[file_name][word] += 1

def create_query_tf_vector(directory_path, feature_list, column_header):
    #Creating TF vectors
    tf_vector_output_file = open(directory_path + "vectors/query_tf_vector.csv", "w")
    csv_write = csv.writer(tf_vector_output_file)
    header = column_header
    csv_write.writerow(header)
    for file_name, words_map in words_in_object_map.items():
        tf_vector = [file_name]
        for feature in column_header[1:]:    
            feature=eval(feature)        
            if feature in words_map:
                tf_vector.append(words_map[feature]/total_count_of_words_in_object[file_name])
            else:
                tf_vector.append(0)
        csv_write.writerow(tf_vector)
    tf_vector_output_file.close()
    print(len(tf_vector))

def create_query_tf_idf_vector(directory_path, words_dir_path, feature_list, column_header):
    number_of_files = get_number_of_gesture_files_in_dir(words_dir_path)
    print(number_of_files)

    words_dir_path = args.output_dir + "words/"
    files = os.listdir(words_dir_path)
    for file_name in files:
        if file_name.endswith(".csv"):
            process_word_file(words_dir_path, file_name)
    #print(words_in_object_map['1_words.csv'])
    count_of_a_feature_in_object = []
    for feature in column_header[1:]:
        count = 0
        feature=eval(feature)
        for file_name, words_map in words_in_object_map.items():
            if feature in words_map:
                count += 1
        count_of_a_feature_in_object.append(count)
    
    #Creating TF-IDF vectors
    tf_vectors = np.array(pd.read_csv(directory_path + "vectors/query_tf_vector.csv"))
    idf_vector = np.array(count_of_a_feature_in_object)
    idf_vector = np.divide(number_of_files, idf_vector)
    idf_vector = np.log(idf_vector)
    # print(idf_vector)
    # max_idf = max(idf_vector)
    # idf_vector = np.divide(idf_vector, max_idf)
    tf_idf_vectors = tf_vectors[0:, 1:] * idf_vector
    tf_idf_vectors = np.hstack((tf_vectors[0:, :1], tf_idf_vectors))

    #adding features as header
    header = column_header
    pd.DataFrame(tf_idf_vectors).to_csv(directory_path + "vectors/query_tf_idf_vector.csv", header = header, index = None)

def similar_gestures(query_vector_path):
    query_vector = np.array(pd.read_csv(args.query_output_dir + query_vector_path))
    dot_products={}
    for v in vectors:
        dot_products[v[0]]=np.dot(query_vector[0][1:],v[1:])
    sort_orders = sorted(dot_products.items(), key=lambda x: x[1], reverse=True)
    for i in range(0,10):
            print(sort_orders[i])

def similar_distance(vectors, query_latent_vector):       
    distances={}
    for v in vectors:
        distances[v[0]]=distance.euclidean(query_latent_vector[0],v[1:])
    sort_orders = sorted(distances.items(), key=lambda x: x[1])
    for i in range(0,10):
        print(sort_orders[i])

def get_sequences(file_path, type):
    sequences = {}
    words = np.array(pd.read_csv(file_path, header = None))
    for row in words:
        if (row[0], row[2]) not in sequences:
            sequences[(row[0], row[2])] = []
        if type == "edit":
            sequences[(row[0], row[2])].append(tuple(row[6:]))
        elif type == "dtw":
            sequences[(row[0], row[2])].append(row[5])
    return sequences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--query_word_file_name', help='Query word file name', required=True)
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--query_output_dir', help='query output directory', required=True)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', required=True)
    args = parser.parse_args()

    vectors_df = pd.read_csv(args.output_dir + "vectors/" + args.vector_model + "_vectors.csv")
    vectors = np.array(vectors_df)

    words_dir_path = args.output_dir + "words/"
    query_words_dir_path = args.query_output_dir + "words/"
    process_word_file(query_words_dir_path, args.query_word_file_name)     

    feature_list = list(feature_dict)
    feature_list.sort()
    create_query_tf_vector(args.output_dir, feature_list,vectors_df.columns)

    if args.user_option == "dot_product":
        print("dot product:")    
        if args.vector_model == "tf":
            similar_gestures("vectors/query_tf_vector.csv")
        else:
            create_query_tf_idf_vector(args.query_output_dir, args.output_dir+"words/", feature_list, vectors_df.columns)
            similar_gestures("vectors/query_tf_idf_vector.csv")

    elif args.user_option == "pca" or args.user_option == "svd" or args.user_option == "nmf" or args.user_option == "lda":
        print(args.user_option)
        pca = joblib.load(args.output_dir + args.vector_model + "_" + args.user_option + ".sav")
        query_vector = np.array(pd.read_csv(args.query_output_dir + "vectors/query_" + args.vector_model +"_vector.csv"))
        query_latent_vector = pca.transform(query_vector[0:,1:])
        vectors = np.array(pd.read_csv(args.output_dir + args.vector_model + "_" + args.user_option + "_vectors.csv", header = None))
        similar_distance(vectors, query_latent_vector)

    elif args.user_option == "edit_distance":
        # start_time = time.time()
        distances = {}
        files = os.listdir(words_dir_path)
        query_sequences = get_sequences(query_words_dir_path + args.query_word_file_name, "edit")
        for file_name in files:
            if file_name.endswith(".csv"):
                distances[file_name] = 0
                sequences = get_sequences(words_dir_path + file_name, "edit")
                for component in ['W', 'X', 'Y', 'Z']:
                    for sensor_id in range(NUM_SENSORS):
                        distances[file_name] += get_edit_distance(sequences[(component, sensor_id)], query_sequences[(component, sensor_id)])

        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        for i in range(0,10):
                print(sorted_distances[i])
        # print(time.time() - start_time)

    elif args.user_option == "dtw":
        distances = {}
        files = os.listdir(words_dir_path)
        query_sequences = get_sequences(query_words_dir_path + args.query_word_file_name, "dtw")
        for file_name in files:
            distances[file_name] = 0
            if file_name.endswith(".csv"):
                sequences = get_sequences(words_dir_path + file_name, "dtw")
                for component in ['W', 'X', 'Y', 'Z']:
                    for sensor_id in range(NUM_SENSORS):
                        distances[file_name] += get_dtw_distance(sequences[(component, sensor_id)], query_sequences[(component, sensor_id)])

        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        for i in range(0,10):
                print(sorted_distances[i])
