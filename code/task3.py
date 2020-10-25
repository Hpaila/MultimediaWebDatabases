import argparse
import os
import numpy as np
import pandas as pd
import csv
import glob
import joblib
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.spatial import distance
from sequence_utils import get_edit_distance, get_dtw_distance
import time
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

NUM_SENSORS = 20

def get_number_of_gesture_files_in_dir(directory_path):
    return len(glob.glob1(directory_path,"*.csv"))

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
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--user_option', help='user option', required=True)
    parser.add_argument('--p', type=int, help='Number of latent components', required=True)
    parser.add_argument('--type', help='svd or nmf', required=True)
    args = parser.parse_args()

    similarity_matrix = None

    similarity_matrix_size = get_number_of_gesture_files_in_dir(args.output_dir + "words/")
    gesture_names = None

    # a = np.array([[1,2],[3,4],[5,6]])
    # print(a.dot(a.T))

    #Get the similarity matrix based on the user option
    if args.user_option == "dot_product":
        vectors = np.array(pd.read_csv(args.output_dir + "vectors/" + args.vector_model + "_vectors.csv"))
        similarity_matrix = vectors[0:, 1:].dot(vectors[0:, 1:].T)
        gesture_names = np.squeeze(vectors[0:,:1].T).tolist()
        
    elif args.user_option == "pca" or args.user_option == "svd" or args.user_option == "nmf" or args.user_option == "lda":
        vectors = np.array(pd.read_csv(args.output_dir + args.vector_model + "_" + args.user_option + "_vectors.csv", header = None))
        # distance_matrix = distance_matrix(vectors[0:,1:], vectors[0:, 1:])
        #TODO change this to euclidean if nan values
        distance_matrix = cdist(vectors[0:,1:], vectors[0:, 1:], metric = "mahalanobis")
        similarity_matrix = 1 / (1 + distance_matrix)
        gesture_names = np.squeeze(vectors[0:,:1].T).tolist()
        
    elif args.user_option == "edit_distance":
        similarity_matrix = [[-1 for x in range(similarity_matrix_size)] for x in range(similarity_matrix_size)] 
        words_dir_path = args.output_dir + "words/"
        files = os.listdir(words_dir_path)
        gesture_names = files
        count = 0
        for file_name1 in files:
            for file_name2 in files:
                row = gesture_names.index(file_name1)
                col = gesture_names.index(file_name2)
                if similarity_matrix[row][col] == -1:
                    similarity_matrix[row][col] = 0
                    count += 1
                    if file_name1.endswith(".csv") and file_name2.endswith(".csv"):
                        sequences1 = get_sequences(words_dir_path + file_name1, "edit")
                        sequences2 = get_sequences(words_dir_path + file_name2, "edit")
                        for component in ['W', 'X', 'Y', 'Z']:
                            for sensor_id in range(NUM_SENSORS):
                                similarity_matrix[row][col] += get_edit_distance(sequences1[(component, sensor_id)], sequences2[(component, sensor_id)])
                        # print(similarity_matrix[row][col])
                        similarity_matrix[col][row] = similarity_matrix[row][col]
        similarity_matrix = np.array(similarity_matrix)
        similarity_matrix = 1 / (1 + similarity_matrix)

    elif args.user_option == "dtw":
        similarity_matrix = [[-1 for x in range(similarity_matrix_size)] for x in range(similarity_matrix_size)] 
        words_dir_path = args.output_dir + "words/"
        files = os.listdir(words_dir_path)
        gesture_names = files
        count = 0
        for file_name1 in files:
            for file_name2 in files:
                row = gesture_names.index(file_name1)
                col = gesture_names.index(file_name2)
                if similarity_matrix[row][col] == -1:
                    similarity_matrix[row][col] = 0
                    count += 1
                    if file_name1.endswith(".csv") and file_name2.endswith(".csv"):
                        sequences1 = get_sequences(words_dir_path + file_name1, "dtw")
                        sequences2 = get_sequences(words_dir_path + file_name2, "dtw")
                        for component in ['W', 'X', 'Y', 'Z']:
                            for sensor_id in range(NUM_SENSORS):
                                similarity_matrix[row][col] += get_dtw_distance(sequences1[(component, sensor_id)], sequences2[(component, sensor_id)])
                        similarity_matrix[col][row] = similarity_matrix[row][col]
        similarity_matrix = np.array(similarity_matrix)
        similarity_matrix = 1 / (1 + similarity_matrix)

    # print(similarity_matrix)
    #TODO do we enable normalization or not??
    # similarity_matrix = np.divide(similarity_matrix - similarity_matrix.min(), similarity_matrix.max() - similarity_matrix.min(), out = similarity_matrix)
    similarity_matrix_with_headers = np.hstack((np.array(gesture_names).reshape(-1,1), similarity_matrix))
    header = gesture_names
    header.insert(0, "Nothing")
    pd.DataFrame(similarity_matrix_with_headers).to_csv(args.output_dir + "similarity_matrix_" + str(args.user_option)+ ".csv", header = gesture_names, index = None)

    #Perform dimensionality reduction on the similarity matrix and save the top p latent gestures
    if args.type == "svd":
        svd = TruncatedSVD(n_components=args.p)
        latent_gestures = svd.fit_transform(similarity_matrix)
        top_p_latent_gestures_scores = open(args.output_dir + "top_p_latent_gestures_scores_svd_" + str(args.user_option), "w")
        
        for row in svd.components_:
            zipped = sorted(zip(row, gesture_names), reverse=True)
            str_zipped = [str(tup) for tup in zipped]
            top_p_latent_gestures_scores.write(",".join(str_zipped) + "\n")
        top_p_latent_gestures_scores.close()

    elif args.type == "nmf":
        nmf = NMF(n_components=args.p)
        latent_gestures = nmf.fit_transform(similarity_matrix)
        top_p_latent_gestures_scores = open(args.output_dir + "top_p_latent_gestures_scores_nmf_" + str(args.user_option), "w")
        
        for row in nmf.components_:
            zipped = sorted(zip(row, gesture_names), reverse=True)
            str_zipped = [str(tup) for tup in zipped]
            top_p_latent_gestures_scores.write(",".join(str_zipped) + "\n")
        top_p_latent_gestures_scores.close()       
        