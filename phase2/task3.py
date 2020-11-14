import argparse
import os
import numpy as np
import pandas as pd
import csv
import glob
import joblib
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.spatial import distance
from phase2.sequence_utils import get_edit_distance, get_dtw_distance
import time
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

NUM_SENSORS = 20
vector_model = "tf"
output_dir = "outputs/"
user_option = "pca"
no_of_latent_features = 4  # value of p
dimensionality_reduction_algo = "svd"  # type
custom_cost = False
gesture_names = None
similarity_matrix = None


def get_number_of_gesture_files_in_dir(directory_path):
    return len(glob.glob1(directory_path, "*.csv"))


def get_sequences(file_path, type):
    sequences = {}
    words = np.array(pd.read_csv(file_path, header=None))
    for row in words:
        if (row[0], row[2]) not in sequences:
            sequences[(row[0], row[2])] = []
        if type == "edit":
            sequences[(row[0], row[2])].append(tuple(row[6:]))
        elif type == "dtw":
            sequences[(row[0], row[2])].append(row[5])
    return sequences


def create_similarity_matrix():
    global similarity_matrix, gesture_names

    similarity_matrix_size = get_number_of_gesture_files_in_dir(output_dir + "words/")

    # Get the similarity matrix based on the user option
    if user_option == "dot_product":
        vectors = np.array(pd.read_csv(output_dir + "vectors/" + vector_model + "_vectors.csv"))
        similarity_matrix = vectors[0:, 1:].dot(vectors[0:, 1:].T)
        gesture_names = np.squeeze(vectors[0:, :1].T).tolist()

    elif user_option == "pca" or user_option == "svd" or user_option == "nmf" or user_option == "lda":
        vectors = np.array(pd.read_csv(output_dir + vector_model + "_" + user_option + "_vectors.csv", header=None))
        # distance_matrix = distance_matrix(vectors[0:,1:], vectors[0:, 1:])
        # TODO change this to euclidean if nan values
        distance_matrix = cdist(vectors[0:, 1:], vectors[0:, 1:], metric="mahalanobis")
        if np.isnan(distance_matrix).any():
            print("distance matrix contains nan values, so calculating using euclidean metric instead of mahalanobis")
            distance_matrix = cdist(vectors[0:, 1:], vectors[0:, 1:], metric="euclidean")
        similarity_matrix = 1 / (1 + distance_matrix)
        gesture_names = np.squeeze(vectors[0:, :1].T).tolist()

    elif user_option == "edit_distance":
        similarity_matrix = [[-1 for x in range(similarity_matrix_size)] for x in range(similarity_matrix_size)]
        words_dir_path = output_dir + "words/"
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
                                similarity_matrix[row][col] += get_edit_distance(sequences1[(component, sensor_id)],
                                                                                 sequences2[(component, sensor_id)],
                                                                                 custom_cost=custom_cost)
                        # print(similarity_matrix[row][col])
                        similarity_matrix[col][row] = similarity_matrix[row][col]
        similarity_matrix = np.array(similarity_matrix)
        similarity_matrix = 1 / (1 + similarity_matrix)

    elif user_option == "dtw":
        similarity_matrix = [[-1 for x in range(similarity_matrix_size)] for x in range(similarity_matrix_size)]
        words_dir_path = output_dir + "words/"
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
                                similarity_matrix[row][col] += get_dtw_distance(sequences1[(component, sensor_id)],
                                                                                sequences2[(component, sensor_id)])
                        similarity_matrix[col][row] = similarity_matrix[row][col]
        similarity_matrix = np.array(similarity_matrix)
        similarity_matrix = 1 / (1 + similarity_matrix)

    # print(similarity_matrix)
    # TODO do we enable normalization or not??
    # similarity_matrix = np.divide(similarity_matrix - similarity_matrix.min(), similarity_matrix.max() - similarity_matrix.min(), out = similarity_matrix)
    similarity_matrix_with_headers = np.hstack((np.array(gesture_names).reshape(-1, 1), similarity_matrix))
    header = list(gesture_names)  # deep copy of the list
    header.insert(0, "Nothing")
    pd.DataFrame(similarity_matrix_with_headers).to_csv(
        output_dir + "similarity_matrix_" + str(user_option) + ".csv", header=header, index=None)
    print("Saving gesture gesture similarity matrix")


def perform_dimensionality_reduction():
    global gesture_names, similarity_matrix

    # Perform dimensionality reduction on the similarity matrix and save the top p latent gestures
    if dimensionality_reduction_algo == "svd":
        svd = TruncatedSVD(n_components=no_of_latent_features)
        latent_gestures = svd.fit_transform(similarity_matrix)
        top_p_latent_gestures_scores = open(output_dir + "top_p_latent_gestures_scores_svd_" + str(user_option), "w")

        for row in svd.components_:
            zipped = sorted(zip(row, gesture_names), reverse=True)
            str_zipped = [str(tup) for tup in zipped]
            top_p_latent_gestures_scores.write(",".join(str_zipped) + "\n")
        top_p_latent_gestures_scores.close()

    elif type == "nmf":
        nmf = NMF(n_components=no_of_latent_features)
        latent_gestures = nmf.fit_transform(similarity_matrix)
        top_p_latent_gestures_scores = open(output_dir + "top_p_latent_gestures_scores_nmf_" + str(user_option), "w")

        for row in nmf.components_:
            zipped = sorted(zip(row, gesture_names), reverse=True)
            str_zipped = [str(tup) for tup in zipped]
            top_p_latent_gestures_scores.write(",".join(str_zipped) + "\n")
        top_p_latent_gestures_scores.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--user_option', help='user option', required=True)
    parser.add_argument('--p', type=int, help='Number of latent components', required=True)
    parser.add_argument('--type', help='svd or nmf', required=True)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', required=False)
    args = parser.parse_args()

    vector_model = args.vector_model
    output_dir = args.output_dir
    user_option = args.user_option
    no_of_latent_features = args.p
    dimensionality_reduction_algo = args.type
    custom_cost = args.custom_cost

    create_similarity_matrix()
    perform_dimensionality_reduction()


def call_task3(local_vector_model, local_output_dir, local_user_option, local_no_of_latent_features,
               local_dimensionality_reduction_algo, local_custom_cost):
    global vector_model, output_dir, user_option, no_of_latent_features, dimensionality_reduction_algo, custom_cost
    vector_model = local_vector_model or vector_model
    output_dir = local_output_dir or output_dir
    user_option = local_user_option or user_option
    no_of_latent_features = local_no_of_latent_features or no_of_latent_features
    dimensionality_reduction_algo = local_dimensionality_reduction_algo or dimensionality_reduction_algo
    custom_cost = local_custom_cost or custom_cost
    create_similarity_matrix()
