import argparse
import numpy as np
import pandas as pd
from copy import deepcopy

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.spatial import distance

def partition_gestures_into_p_groups(svd_or_nmf_type):
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
    print(threshold)
    
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
    km = KMeans(init="random", n_clusters=num_of_clusters)
    km.fit(similarity_matrix)
    return km.labels_
    

def k_means_clustering(data, number_of_clusters):
    n = data.shape[0]  # Number of training data
    c = data.shape[1]  # Number of features in the data

    centers_old_index = np.random.randint(n, size=number_of_clusters)
    for i in range(number_of_clusters):
        centers_old = data[centers_old_index]

    centers_new = np.zeros(centers_old.shape)
    distances = np.zeros((n, number_of_clusters))

    error = np.linalg.norm(centers_new - centers_old)
    prev_error = None
    for i in range(95):
        prev_error = error
        for i in range(number_of_clusters):
            distances[:, i] = np.linalg.norm(data - centers_old[i], axis = 1)
        clusters = np.argmin(distances, axis = 1)
        # print("clusters ", clusters)
        centers_old = deepcopy(centers_new)
        for i in range(number_of_clusters):
            if len(data[clusters == i]) != 0:
                centers_new[i] = np.mean(data[clusters == i], axis = 0)
            else:
                centers_new[i] = np.zeros(c)
        error = np.linalg.norm(centers_new - centers_old)
    # print(clusters)
    return clusters

def kmeans(X,k=3,max_iterations=100):
    '''
    X: multidimensional data
    k: number of clusters
    max_iterations: number of repetitions before clusters are established
    
    Steps:
    1. Convert data to numpy aray
    2. Pick indices of k random point without replacement
    3. Find class (P) of each data point using euclidean distance
    4. Stop when max_iteration are reached of P matrix doesn't change
    
    Return:
    np.array: containg class of each data point
    '''
    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[P==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
        if np.array_equal(P,tmp):break
        P = tmp
    return P
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--user_option', help='user option', required=True)
    parser.add_argument('--p', type=int, help='Number of clusters', required=True)
    args = parser.parse_args()

    # Task 4a - Using SVD from 3a
    partitions = partition_gestures_into_p_groups("svd")
    for key in partitions:
        print("Latent feature ", key)
        print("---------------------------------------------------")
        print(partitions[key])
        print()
    # Task 4b - Using NMF from 3b
    # partitions = print(partition_gestures_into_p_groups("nmf"))

    # Laplacian spectral clustering
    similarity_matrix_with_headers = np.array(pd.read_csv(args.output_dir + "similarity_matrix_" + args.user_option + ".csv", header=None))
    gestures_list = similarity_matrix_with_headers[0][1:]
    similarity_matrix = np.array(similarity_matrix_with_headers[1:, 1:], dtype=float)
    clusters = perform_laplacian_spectral_clustering(similarity_matrix, args.p)
    # clusters = k_means_clustering(similarity_matrix, args.p) 

    # clusters = kmeans(similarity_matrix, k = args.p)

    for i in range(args.p):
        print("\nCluster ", i)
        print("---------------------------------------------------")
        indexes = np.where(clusters == i)[0]
        for index in indexes:
            print(gestures_list[index])
        print()
    
    # km = KMeans(init="random", n_clusters=args.p)
    # km.fit(similarity_matrix)
    # print(km.labels_)

    # for i in range(args.p):
    #     print("\nCluster ", i)
    #     print("---------------------------------------------------")
    #     indexes = np.where(km.labels_ == i)[0]
    #     for index in indexes:
    #         print(gestures_list[index])
    #     print()
