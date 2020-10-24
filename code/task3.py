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

def get_number_of_gesture_files_in_dir(directory_path):
    return len(glob.glob1(directory_path,"*.csv"))

def get_sequences(file_path, type):
    sequences = {}
    words = np.array(pd.read_csv(file_path, header = None))
    for row in words:
        if row[0] not in sequences:
            sequences[row[0]] = []
        if type == "edit":
            sequences[row[0]].append(tuple(row[6:]))
        elif type == "dtw":
            sequences[row[0]].append(row[5])
    return sequences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--user_option', type=int, help='user option', required=True)
    parser.add_argument('--p', type=int, help='Number of latent components', required=True)
    parser.add_argument('--type', help='svd or nmf', required=True)
    args = parser.parse_args()

    similarity_matrix = None
    user_option_map = {2: "pca", 3: "svd", 4: "nmf", 5: "lda"}

    similarity_matrix_size = get_number_of_gesture_files_in_dir(args.output_dir + "words/")
    gesture_names = None

    # a = np.array([[1,2],[3,4],[5,6]])
    # print(a.dot(a.T))

    #Get the similarity matrix based on the user option
    if args.user_option == 1:
        vectors = np.array(pd.read_csv(args.output_dir + "vectors/" + args.vector_model + "_vectors.csv"))
        similarity_matrix = vectors[0:, 1:].dot(vectors[0:, 1:].T)
        gesture_names = np.squeeze(vectors[0:,:1].T).tolist()

    elif args.user_option == 2 or args.user_option == 3 or args.user_option == 4 or args.user_option == 5:
        vectors = np.array(pd.read_csv(args.output_dir + args.vector_model + "_" + user_option_map[args.user_option] + "_vectors.csv", header = None))
        distance_matrix = distance_matrix(vectors[0:,1:], vectors[0:, 1:])
        #TODO Need to change this to similarity matrix
        similarity_matrix = distance_matrix
        gesture_names = np.squeeze(vectors[0:,:1].T).tolist()

    elif args.user_option == 6:
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
                    count += 1
                    if file_name1.endswith(".csv") and file_name2.endswith(".csv"):
                        sequences1 = get_sequences(words_dir_path + file_name1, "edit")
                        sequences2 = get_sequences(words_dir_path + file_name2, "edit")
                        similarity_matrix[row][col] = get_edit_distance(sequences1["W"], sequences2["W"]) + get_edit_distance(sequences1["X"], sequences2["X"]) + get_edit_distance(sequences1["Y"], sequences2["Y"]) + get_edit_distance(sequences1["Z"], sequences2["Z"])
                        similarity_matrix[col][row] = similarity_matrix[row][col]
    elif args.user_option == 7:
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
                    count += 1
                    if file_name1.endswith(".csv") and file_name2.endswith(".csv"):
                        sequences1 = get_sequences(words_dir_path + file_name1, "dtw")
                        sequences2 = get_sequences(words_dir_path + file_name2, "dtw")
                        similarity_matrix[row][col] = get_dtw_distance(sequences1["W"], sequences2["W"]) + get_dtw_distance(sequences1["X"], sequences2["X"]) + get_dtw_distance(sequences1["Y"], sequences2["Y"]) + get_dtw_distance(sequences1["Z"], sequences2["Z"])
                        similarity_matrix[col][row] = similarity_matrix[row][col]
    
    #Perform dimensionality reduction on the similarity matrix and save the top p latent gestures
    if args.type == "svd":
        svd = TruncatedSVD(n_components=args.p)
        latent_gestures = svd.fit_transform(similarity_matrix)
        top_p_latent_gestures_scores = open(args.output_dir + "top_p_latent_gestures_scores_svd_" + str(args.user_option), "w")
        
        for row in svd.components_:
            zipped = sorted(zip(row, gesture_names), reverse=True)
            top_p_latent_gestures_scores.write(str(zipped) + "\n")
        top_p_latent_gestures_scores.close()

    elif args.type == "nmf":
        nmf = NMF(n_components=args.p)
        latent_gestures = nmf.fit_transform(similarity_matrix)
        top_p_latent_gestures_scores = open(args.output_dir + "top_p_latent_gestures_scores_nmf_" + str(args.user_option), "w")
        
        for row in nmf.components_:
            zipped = sorted(zip(row, gesture_names), reverse=True)
            top_p_latent_gestures_scores.write(str(zipped) + "\n")
        top_p_latent_gestures_scores.close()       
        