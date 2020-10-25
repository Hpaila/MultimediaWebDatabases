import argparse
import os
import numpy as np
import pandas as pd
import csv
import glob
import joblib
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation, NMF
from scipy.spatial import distance

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

def create_tf_vectors(directory_path, feature_list, column_header):
    #Creating TF vectors
    tf_vector_output_file = open(directory_path + "vectors/tf_vectors_2.csv", "w")
    csv_write = csv.writer(tf_vector_output_file)
    header = column_header
    csv_write.writerow(header)
    for file_name, words_map in words_in_object_map.items():
        tf_vector = [file_name]
        for feature in column_header[1:]:    
            feature=eval(feature)        
            if feature in words_map:
                tf_vector.append(words_map[feature]/total_count_of_words_in_object[file_name])
            else:
                tf_vector.append(0)
        csv_write.writerow(tf_vector)
    tf_vector_output_file.close()
    print(len(tf_vector))


def create_tf_idf_vectors(directory_path, gestures_dir, feature_list,column_header):
    number_of_files = get_number_of_gesture_files_in_dir(gestures_dir + "W/")
    print(number_of_files)

    words_dir_path = args.output_dir + "words/"
    files = os.listdir(words_dir_path)
    for file_name in files:
        if file_name.endswith(".csv"):
            process_word_file(words_dir_path, file_name)
    #print(words_in_object_map['1_words.csv'])
    count_of_a_feature_in_object = []
    for feature in column_header[1:]:
        count = 0
        feature=eval(feature)
        for file_name, words_map in words_in_object_map.items():
            if feature in words_map:
                count += 1
        count_of_a_feature_in_object.append(count)
    
    #Creating TF-IDF vectors
    tf_vectors = np.array(pd.read_csv(directory_path + "vectors/tf_vectors_2.csv"))
    idf_vector = np.array(count_of_a_feature_in_object)
    idf_vector = np.divide(number_of_files, idf_vector)
    idf_vector = np.log(idf_vector)
    # print(idf_vector)
    # max_idf = max(idf_vector)
    # idf_vector = np.divide(idf_vector, max_idf)
    tf_idf_vectors = tf_vectors[0:, 1:] * idf_vector
    tf_idf_vectors = np.hstack((tf_vectors[0:, :1], tf_idf_vectors))

    #adding features as header
    header = column_header
    pd.DataFrame(tf_idf_vectors).to_csv(directory_path + "vectors/tf_idf_vectors_2.csv", header = header, index = None)

def similar_gestures(csv):
            gesture_vector_df = pd.read_csv(args.output_dir + csv)
            gesture_vector = np.array(gesture_vector_df)
            dot_products={}
            for v in vectors:
                dot_products[v[0]]=np.dot(gesture_vector[0][1:],v[1:])
            sort_orders = sorted(dot_products.items(), key=lambda x: x[1], reverse=True)
            for i in range(0,10):
                    print(sort_orders[i])

def similar_distance(vectors):       
        distances={}
        for v in vectors:
                distances[v[0]]=distance.euclidean(latent_vectors[0],v[1:])
        sort_orders = sorted(distances.items(), key=lambda x: x[1])
        for i in range(0,10):
                    print(sort_orders[i])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vector models.')
    parser.add_argument('--gesture', help='gesture file', required=True)
    parser.add_argument('--vector_model', help='vector model', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--type', type=int, help='Type of dimensionality reduction', required=True)
    parser.add_argument('--gestures_dir', help='gestures directory', required=True)  
    args = parser.parse_args()
    
    vectors_df = None   
    if args.vector_model == "tf":
        vectors_df = pd.read_csv(args.output_dir + "vectors/tf_vectors.csv")
        csv="vectors/tf_vectors_2.csv"
        #print(len(vectors_df.columns))
    elif args.vector_model == "tf_idf":
        vectors_df = pd.read_csv(args.output_dir + "vectors/tf_idf_vectors.csv")
        csv="vectors/tf_idf_vectors_2.csv"

    vectors = np.array(vectors_df)

    if args.type == 1:
        print("dot product:")
        words_dir_path = args.output_dir + "words/"
        process_word_file(words_dir_path, args.gesture)        
        #print(len(feature_dict))
        feature_list = list(feature_dict)
        feature_list.sort()
        
        #creating vectors directory
        try:
            os.mkdir(args.output_dir + "vectors/")
        except OSError as error:
            print("vectors directory already exists")

        create_tf_vectors(args.output_dir, feature_list,vectors_df.columns)
        #create_tf_idf_vectors(args.output_dir, args.gestures_dir, feature_list)
        if args.vector_model == "tf":
            similar_gestures("vectors/tf_vectors_2.csv")
            
        else:
            create_tf_idf_vectors(args.output_dir, args.gestures_dir, feature_list,vectors_df.columns)
            similar_gestures("vectors/tf_idf_vectors_2.csv")

    elif args.type == 2:
        print("pca")
        pca = joblib.load(args.output_dir+"pca.sav")
        gesture_vector_df = pd.read_csv(args.output_dir + csv)
        gesture_vector = np.array(gesture_vector_df)
        #print(gesture_vector[0:,1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        latent_vectors = pca.transform(gesture_vector[0:,1:])
        #latent_vectors = np.hstack((gesture_vector[0][0], latent_vectors))
        vectors_df = pd.read_csv(args.output_dir + "pca_vectors.csv")
        vectors = np.array(vectors_df)
        similar_distance(vectors)

    elif args.type==3:
        print("svd")
        pca = joblib.load(args.output_dir+"svd.sav") 
        gesture_vector_df = pd.read_csv(args.output_dir + csv)
        gesture_vector = np.array(gesture_vector_df)
        #print(gesture_vector[0:,1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        latent_vectors = pca.transform(gesture_vector[0:,1:])
        #latent_vectors = np.hstack((gesture_vector[0][0], latent_vectors))
        vectors_df = pd.read_csv(args.output_dir + "svd_vectors.csv")
        vectors = np.array(vectors_df)
        similar_distance(vectors)

    elif args.type==4:
        print("nmf")
        pca = joblib.load(args.output_dir+"nmf.sav") 
        gesture_vector_df = pd.read_csv(args.output_dir + csv)
        gesture_vector = np.array(gesture_vector_df)
        #print(gesture_vector[0:,1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        latent_vectors = pca.transform(gesture_vector[0:,1:])
        #latent_vectors = np.hstack((gesture_vector[0][0], latent_vectors))
        vectors_df = pd.read_csv(args.output_dir + "nmf_vectors.csv")
        vectors = np.array(vectors_df)
        similar_distance(vectors)

    elif args.type==5:
        print("lda")
        pca = joblib.load(args.output_dir+"lda.sav") 
        gesture_vector_df = pd.read_csv(args.output_dir + csv)
        gesture_vector = np.array(gesture_vector_df)
        #print(gesture_vector[0:,1:])
        # saving the model, so that it can be used in future tasks to transform the query gesture
        latent_vectors = pca.transform(gesture_vector[0:,1:])
        #latent_vectors = np.hstack((gesture_vector[0][0], latent_vectors))
        vectors_df = pd.read_csv(args.output_dir + "lda_vectors.csv")
        vectors = np.array(vectors_df)
        similar_distance(vectors)

    
        
        

