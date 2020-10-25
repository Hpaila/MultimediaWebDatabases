import argparse
import numpy as np
import pandas as pd
<<<<<<< HEAD
from copy import deepcopy

=======
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
>>>>>>> task4b partial changes

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

def perform_laplacian_spectral_clustering(similarity_matrix, num_of_clusters, user_option):
    threshold = 0.91
    if user_option == 1:
        threshold = 0.0015

    vectorizer = np.vectorize(lambda x: 1 if x > threshold else 0)
    W = np.vectorize(vectorizer)(similarity_matrix)
    print(W)
    D = np.diag(np.sum(np.array(csr_matrix(W).todense()), axis=1))
    # print('degree matrix:')
    # print(D)
    
    L = D - W
    # print('laplacian matrix:')
    # print(L)

    eigen_values, eigen_vectors = np.linalg.eig(L)

    i = np.where(eigen_values < 0.5)[0]

    km = KMeans(init='k-means++', n_clusters=5)
    km.fit(eigen_vectors)
    print(km.labels_)
    


def k_means_clustering(data, number_of_clusters):
    n = data.shape[0]  # Number of training data
    c = data.shape[1]  # Number of features in the data

    centers_old_index = np.random.randint(n, size=number_of_clusters)
    for i in range(number_of_clusters):
        centers_old = data[centers_old_index]
    centers_new = np.zeros(centers_old.shape)
    distances = np.zeros((n, number_of_clusters))

    while (centers_old != centers_new).all():
        for i in range(number_of_clusters):
            distances[:, i] = np.linalg.norm(data - centers_old[i], axis=1)
        clusters = np.argmin(distances, axis=1)
        print("clusters ", clusters)
        centers_old = deepcopy(centers_new)
        for i in range(number_of_clusters):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
    print(centers_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--user_option', type=int, help='user option', required=True)
    parser.add_argument('--p', type=int, help='Number of clusters', required=True)
    args = parser.parse_args()

    # Task 4a - Using SVD from 3a
    print(partition_gestures_into_p_groups("svd"))
    
    # Task 4b - Using NMF from 3b
    # print(partition_gestures_into_p_groups("nmf"))

    # Laplacian spectral clustering
    similarity_matrix_with_headers = np.array(pd.read_csv(args.output_dir + "similarity_matrix_" + str(args.user_option) + ".csv", header=None))
    similarity_matrix = np.array(similarity_matrix_with_headers[1:, 1:], dtype=float)
    perform_laplacian_spectral_clustering(similarity_matrix, args.p)
    k_means_clustering(similarity_matrix, args.p)

