import argparse
import os
import numpy as np
import pandas as pd
import csv
import glob

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

def create_tf_vectors(directory_path, feature_list):
    #Creating TF vectors
    tf_vector_output_file = open(directory_path + "vectors/tf_vectors.csv", "w")
    csv_write = csv.writer(tf_vector_output_file)
    header = [(-1,-1,-1,-1)]
    header.extend(feature_list)
    csv_write.writerow(header)
    for file_name, words_map in words_in_object_map.items():
        tf_vector = [file_name]
        for feature in feature_list:
            if feature in words_map:
                tf_vector.append(words_map[feature]/total_count_of_words_in_object[file_name])
            else:
                tf_vector.append(0)
        csv_write.writerow(tf_vector)
    tf_vector_output_file.close()

def create_tf_idf_vectors(directory_path, words_dir_path, feature_list):
    number_of_files = get_number_of_gesture_files_in_dir(words_dir_path)
    print(number_of_files)

    count_of_a_feature_in_object = []
    for feature in feature_list:
        count = 0
        for file_name, words_map in words_in_object_map.items():
            if feature in words_map:
                count += 1
        count_of_a_feature_in_object.append(count)
    
    #Creating TF-IDF vectors
    tf_vectors = np.array(pd.read_csv(directory_path + "vectors/tf_vectors.csv"))
    idf_vector = np.array(count_of_a_feature_in_object)
    idf_vector = np.divide(number_of_files, idf_vector)
    idf_vector = np.log(idf_vector)
    # print(idf_vector)
    # max_idf = max(idf_vector)
    # idf_vector = np.divide(idf_vector, max_idf)
    tf_idf_vectors = tf_vectors[0:, 1:] * idf_vector
    tf_idf_vectors = np.hstack((tf_vectors[0:, :1], tf_idf_vectors))

    #adding features as header
    header = [(-1,-1,-1,-1)]
    header.extend(feature_list)
    pd.DataFrame(tf_idf_vectors).to_csv(directory_path + "vectors/tf_idf_vectors.csv", header = header, index = None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--output_dir', help='output directory', required=True)
    args = parser.parse_args()

    words_dir_path = args.output_dir + "words/"
    files = os.listdir(words_dir_path)
    for file_name in files:
        if file_name.endswith(".csv"):
            process_word_file(words_dir_path, file_name)
    
    print(len(feature_dict))

    feature_list = list(feature_dict)
    feature_list.sort()
    
    #creating vectors directory
    try:
        os.mkdir(args.output_dir + "vectors/")
    except OSError as error:
        print("vectors directory already exists")

    create_tf_vectors(args.output_dir, feature_list)
    create_tf_idf_vectors(args.output_dir, words_dir_path, feature_list)