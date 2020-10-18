from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation, NMF
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--k', type=int, help='Number of components', required=True)
    parser.add_argument('--type', type=int, help='Type of dimensionality reduction', required=True)
    args = parser.parse_args()

    vectors_df = None
    if args.vector_model == "tf":
        vectors_df = pd.read_csv(args.output_dir + "vectors/tf_vectors.csv")
    elif args.vector_model == "tf_idf":
        vectors_df = pd.read_csv(args.output_dir + "vectors/tf_idf_vectors.csv")
    else:
        print("Invalid vector model, please enter tf or tf_idf")

    vectors = np.array(vectors_df)
    
    if args.type == 1:
        pca = PCA(n_components=args.k)
        pca.fit(vectors[0:, 1:])

        latent_vectors = pca.transform(vectors[0:, 1:])
        pca_components_scores_output = open(args.output_dir + "pca_component_scores", "w")
        for row in pca.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            pca_components_scores_output.write(str(zipped) + "\n")

        pca_components_scores_output.close()
    elif args.type == 2:
        svd = TruncatedSVD(n_components=args.k)
        svd.fit(vectors[0:, 1:])

        latent_vectors = svd.transform(vectors[0:, 1:])
        svd_components_scores_output = open(args.output_dir + "svd_component_scores", "w")
        for row in svd.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            svd_components_scores_output.write(str(zipped) + "\n")
        svd_components_scores_output.close()
    elif args.type == 3:
        nmf = NMF(n_components=args.k)
        nmf.fit(vectors[0:, 1:])

        latent_vectors = nmf.transform(vectors[0:, 1:])
        nmf_components_scores_output = open(args.output_dir + "nmf_component_scores", "w")
        for row in nmf.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            nmf_components_scores_output.write(str(zipped) + "\n")

        nmf_components_scores_output.close()
    elif args.type == 4:
        lda = LatentDirichletAllocation(n_components=args.k)
        lda.fit(vectors[0:, 1:])

        latent_vectors = lda.transform(vectors[0:, 1:])
        lda_components_scores_output = open(args.output_dir + "lda_component_scores", "w")
        for row in lda.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            lda_components_scores_output.write(str(zipped) + "\n")

        lda_components_scores_output.close()
    else:
        print("Invalid type, please select value from 1 to 4")
        