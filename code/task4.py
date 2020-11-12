import argparse
import numpy as np
import pandas as pd
from copy import deepcopy

from scipy.sparse import csr_matrix
from scipy.spatial import distance
import os

def partition_gestures_into_p_groups(svd_or_nmf_type):
    
    files = os.listdir(args.output_dir)
    fname = "top_p_latent_gestures_scores_" + svd_or_nmf_type + "_" + str(args.user_option)
    if fname not in files :
        print("File not found!\n")
        return []
    
    with open(args.output_dir + "top_p_latent_gestures_scores_" + svd_or_nmf_type + "_" + str(args.user_option)) as f:
        gestures_scores = []
        for line in f:
            # elements = tuple(t(e) for t,e in zip(types, line.split()))
            gestures_scores.append(list(eval(line)))
    
    for latent_feature in gestures_scores:
        latent_feature.sort(key = lambda x: x[1])
    
    partition_map = {}
    for i in range(len(gestures_scores[0])):
        temp = [row[i] for row in gestures_scores]
        max_value = max(temp, key = lambda item: item[0])
        latent_feature_index = temp.index(max_value)
        if latent_feature_index not in partition_map:
            partition_map[latent_feature_index] = []
        partition_map[latent_feature_index].append(max_value[1])
    
    return partition_map

def perform_laplacian_spectral_clustering(similarity_matrix, num_of_clusters):
    threshold = np.median(similarity_matrix)
    
    vectorizer = np.vectorize(lambda x: 1 if x > threshold else 0)
    # creating the adjacency graph matrix using median as threshold
    W = np.vectorize(vectorizer)(similarity_matrix)
    
    # creating the diagonal matrix which contains the degree of each vertex
    D = np.diag(np.sum(np.array(csr_matrix(W).todense()), axis = 1))
    
    # creating the laplacian matrix
    L = D - W

    # generating the eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(L)

    # running kmeans on the generated eign vectors
    return KMeans(eigen_vectors, args.p)


def KMeans(similarity_matrix, num_of_clusters):
    max_iterations = 100
    center_ids = np.random.choice(len(similarity_matrix), num_of_clusters, replace=False)
    centers = similarity_matrix[center_ids, :]
    clusters = np.argmin(distance.cdist(similarity_matrix, centers, 'euclidean'), axis = 1)
    for i in range(max_iterations):
        centers = np.vstack([similarity_matrix[clusters == i, :].mean(axis = 0) for i in range(num_of_clusters)])
        new_clusters = np.argmin(distance.cdist(similarity_matrix, centers, 'euclidean'), axis=1)
        if np.array_equal(clusters, new_clusters):
            break
        clusters = new_clusters
    return clusters
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--user_option', help='user option', required=True)
    parser.add_argument('--p', type=int, help='Number of clusters', required=True)
    args = parser.parse_args()

    # Task 4a - Using SVD from 3a
    print("Partitioning gestures based on degree of membership to the latent feature (SVD)")
    partitions = partition_gestures_into_p_groups("svd")
    for key in partitions:
        print("Latent feature ", key+1)
        print("---------------------------------------------------")
        print(partitions[key])
        print()

    # Task 4b - Using NMF from 3b
    print("Partitioning gestures based on degree of membership to the latent feature (NMF)")
    partitions = partition_gestures_into_p_groups("nmf")
    for key in partitions:
        print("Latent feature ", key+1)
        print("---------------------------------------------------")
        print(partitions[key])
        print()
        
    # Laplacian spectral clustering
    similarity_matrix_with_headers = np.array(pd.read_csv(args.output_dir + "similarity_matrix_" + args.user_option + ".csv", header=None))
    gestures_list = similarity_matrix_with_headers[0][1:]
    similarity_matrix = np.array(similarity_matrix_with_headers[1:, 1:], dtype=float)

    clusters = KMeans(similarity_matrix, args.p)
    print("Clusters using KMeans clustering")
    print("---------------------------------------------------")
    for i in range(args.p):
        print("\nCluster ", i)
        print("---------------------------------------------------")
        indexes = np.where(clusters == i)[0]
        for index in indexes:
            print(gestures_list[index])
        print()

    clusters = perform_laplacian_spectral_clustering(similarity_matrix, args.p)
    print("Clusters using Laplacian clustering")
    print("---------------------------------------------------")
    for i in range(args.p):
        print("\nCluster ", i)
        print("---------------------------------------------------")
        indexes = np.where(clusters == i)[0]
        for index in indexes:
            print(gestures_list[index])
        print()
