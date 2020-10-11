import pandas as pd
import numpy as np
import scipy.stats
import csv
import argparse
import os
from statistics import mean

MEAN = 0  # from Phase 1
STANDARD_DEVIATION = 0.25  # from Phase 1
OUTPUT_FOLDER = "./Outputs"
OUTPUT_FILE_PATH = "./Outputs/words"


def task1():
    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')
    parser.add_argument('--dir', help='directory of input data', required=True)
    parser.add_argument('--window', type=int, help='window length', required=True)
    parser.add_argument('--shift', type=int, help='shift length', required=True)
    parser.add_argument('--resolution', type=int, help='resolution', required=True)

    args = parser.parse_args()

    if not os.path.isdir(OUTPUT_FOLDER):
        try:
            os.mkdir(OUTPUT_FOLDER)
        except OSError:
            print("Creation of the directory %s failed, because it already exists" % OUTPUT_FOLDER)
        else:
            print("Successfully created the directory %s" % OUTPUT_FOLDER)

    dir_list = os.listdir(args.dir)
    for directory in dir_list:
        if directory in ['X']:
            component_id = directory
            files = os.listdir(args.dir+directory)
            print(files)
            for file in files:
                if ".csv" in file:
                    sensor_data = pd.DataFrame(pd.read_csv(args.dir + directory + '/'+ file, header=None)).to_numpy()
                    data_avg = np.mean(sensor_data, axis=1)
                    # print(data_avg)
                    data_std = np.std(sensor_data, axis=1)
                    # print(data_std)
                    normalized_data = normalize(sensor_data)
                    # print(normalized_data.shape)
                    quantized_data = gaussian_transform(pd.DataFrame(normalized_data), get_gaussian_bands(args.resolution), args.resolution)
                    # print(sensor_quantized_data)
                    get_window_avg(normalized_data, args.window, args.shift)
                    get_quantized_window(quantized_data, args.window, args.shift)
                    break
                    #         write_words(quantized_data, args.shift, args.window, f)

def get_quantized_window(data, window, shift):
    for i, row in enumerate(data):
        h = 0
        while h + window < len(row):
            # print("h = ", h, " window values ", row[h:h+window])
            temp = [i, h, row[h:h+window].tolist()]
            h += shift
            print(temp)
        break

def get_window_avg(data, window, shift):
    for i, row in enumerate(data):
        h = 0
        while h + window < len(row):
            # print("h = ", h, " window values ", row[h:h+window])
            val = row[h:h+window].tolist()
            temp = [i, h, mean(val)]
            h += shift
            print(temp)
        break


def write_words(d, shift, window, file):
    file_no = file.replace(".csv", "")
    data = d.to_numpy()
    max_time = len(data[0])
    max_sensors = len(data)
    sensor = 0
    if not os.path.isdir(OUTPUT_FILE_PATH):
        try:
            os.mkdir(OUTPUT_FILE_PATH)
        except OSError:
            print("Creation of the directory %s failed, because it already exists" % OUTPUT_FILE_PATH)
        else:
            print("Successfully created the directory %s" % OUTPUT_FILE_PATH)
    file_name = OUTPUT_FILE_PATH + "/" + file_no + "words.csv"
    with open(file_name, mode='w') as output:
        fp = csv.writer(output, delimiter=',')
        while sensor < max_sensors:
            for t in range(0, max_time, shift):
                row = [file_no, sensor, t]
                row += data[sensor][t:t + window].tolist()
                if len(row) == 3 + window:
                    # prune words which do not have enough data
                    fp.writerow(row)
            sensor += 1


def normalize(data):
    data = np.transpose(data)
    new_data = np.divide(data - data.min(axis=0), data.max(axis=0) - data.min(axis=0), out=data,
                         where=(data.max(axis=0) - data.min(axis=0)) != 0)
    new_data = (new_data * 2) - 1
    new_data = np.transpose(new_data)
    return new_data


def get_gaussian_bands(resolution):
    r = resolution
    gaussian_bands = [-1]
    for i in range(1, 2 * r + 1):
        lower_bound = (i - r - 1) / r
        upper_bound = (i - r) / r
        gaussian_bands.append(gaussian_bands[-1] + 2 * (scipy.stats.norm(MEAN, STANDARD_DEVIATION).cdf(upper_bound) -
                                                        scipy.stats.norm(MEAN, STANDARD_DEVIATION).cdf(lower_bound)))
    gaussian_bands[-1] = 1
    return gaussian_bands


def gaussian_transform(data, bands, resolution):
    def get_band_number(x):
        for i in range(len(bands) - 1):
            if bands[i] <= x <= bands[i + 1]:
                return i + 1
        if x > bands[-1]:
            return 2 * resolution

    return data.transform([get_band_number])


if __name__ == '__main__':
    task1()
