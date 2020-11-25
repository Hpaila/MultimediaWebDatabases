import pandas as pd
import numpy as np
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computes accuracy for classified data')
    parser.add_argument('--truth', help='truth labels', default="Phase3_data_for_report/all_labels.csv", required=False)
    parser.add_argument('--file', help='classified data', default="outputs/ppr_2_classification.csv", required=False)

    args = parser.parse_args()

    truth_matrix = np.array(pd.read_csv(args.truth, header=None))
    data_matrix = np.array(pd.read_csv(args.file, header=None))

    data_dict = {}
    for row in truth_matrix:
        data_dict[str(row[0])] = row[1]
    no_of_correct_labels = 0

    for row in data_matrix:
        if data_dict[str(row[0])] == row[1]:
            no_of_correct_labels += 1

    accuracy = no_of_correct_labels/len(data_matrix)
    print(no_of_correct_labels)
    print("ACCURACY for ", args.file, " is ", accuracy)