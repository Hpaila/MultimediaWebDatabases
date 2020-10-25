import argparse
import numpy as np
import pandas as pd

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
    pass

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
    similarity_matrix = similarity_matrix_with_headers[1:, 1:]
    perform_laplacian_spectral_clustering(similarity_matrix, args.p)