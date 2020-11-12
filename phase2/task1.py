from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation, NMF
import pandas as pd
import numpy as np
import argparse
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--k', type=int, help='Number of latent semantics', required=True)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', required=True)
    args = parser.parse_args()
    
    vectors_df = pd.read_csv(args.output_dir + "vectors/" + args.vector_model + "_vectors.csv")
    vectors = np.array(vectors_df)
    
    if args.user_option == "pca":
        pca = PCA(n_components=args.k)
        pca.fit(vectors[0:, 1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        joblib.dump(pca, args.output_dir + args.vector_model + "_pca.sav")
        latent_vectors = pca.transform(vectors[0:, 1:])
        latent_vectors = np.hstack((vectors[0:, :1], latent_vectors))
        pca_components_scores_output = open(args.output_dir + args.vector_model + "_pca_component_scores" , "w")
        for row in pca.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            pca_components_scores_output.write(str(zipped) + "\n")

        pca_components_scores_output.close()
        pd.DataFrame(latent_vectors).to_csv(args.output_dir + args.vector_model + "_pca_vectors.csv", header = None, index = None)
    elif args.user_option == "svd":
        svd = TruncatedSVD(n_components=args.k)
        svd.fit(vectors[0:, 1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        joblib.dump(svd, args.output_dir + args.vector_model + "_svd.sav")

        latent_vectors = svd.transform(vectors[0:, 1:])
        latent_vectors = np.hstack((vectors[0:, :1], latent_vectors))
        svd_components_scores_output = open(args.output_dir + args.vector_model + "_svd_component_scores", "w")
        for row in svd.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            svd_components_scores_output.write(str(zipped) + "\n")
        svd_components_scores_output.close()
        pd.DataFrame(latent_vectors).to_csv(args.output_dir + args.vector_model + "_svd_vectors.csv", header = None, index = None)
    elif args.user_option == "nmf":
        nmf = NMF(n_components=args.k)
        nmf.fit(vectors[0:, 1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        joblib.dump(nmf, args.output_dir + args.vector_model + "_nmf.sav")

        latent_vectors = nmf.transform(vectors[0:, 1:])
        latent_vectors = np.hstack((vectors[0:, :1], latent_vectors))
        nmf_components_scores_output = open(args.output_dir + args.vector_model + "_nmf_component_scores", "w")
        for row in nmf.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            nmf_components_scores_output.write(str(zipped) + "\n")

        nmf_components_scores_output.close()
        pd.DataFrame(latent_vectors).to_csv(args.output_dir + args.vector_model + "_nmf_vectors.csv", header = None, index = None)
    elif args.user_option == "lda":
        lda = LatentDirichletAllocation(n_components=args.k)
        lda.fit(vectors[0:, 1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        joblib.dump(lda, args.output_dir + args.vector_model + "_lda.sav")

        latent_vectors = lda.transform(vectors[0:, 1:])
        latent_vectors = np.hstack((vectors[0:, :1], latent_vectors))
        lda_components_scores_output = open(args.output_dir + args.vector_model + "_lda_component_scores", "w")
        for row in lda.components_:
            zipped = sorted(zip(row, vectors_df.columns[1:]), reverse=True)
            lda_components_scores_output.write(str(zipped) + "\n")

        lda_components_scores_output.close()
        pd.DataFrame(latent_vectors).to_csv(args.output_dir + args.vector_model + "_lda_vectors.csv", header = None, index = None)
    else:
        print("Invalid user option, please enter one of pca, svd, nmf, lda")
        