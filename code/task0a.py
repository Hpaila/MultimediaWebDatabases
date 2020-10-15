import pandas as pd
import numpy as np
import scipy.stats
import csv
import argparse
import os
from statistics import mean

MEAN = 0  # from Phase 1
STANDARD_DEVIATION = 0.25  # from Phase 1
OUTPUT_FOLDER = "./outputs"
OUTPUT_FILE_PATH = "./outputs/words"
band_ranges = []
feature_dict = set()

def getBandNumber(normalized_value):
    for i in range(len(band_ranges)):
        if normalized_value >= band_ranges[i][0] and normalized_value <= band_ranges[i][1]:
            return i+1
    return -1

def generate_band_ranges(r):
    start = -1.0
    total_area = scipy.stats.norm(0, 0.25).cdf(1) - scipy.stats.norm(0, 0.25).cdf(-1)
    for i in range(1, 2*r+1):
        lower_bound = (i-r-1)/r
        upper_bound = (i-r)/r
        len1 = (scipy.stats.norm(0, 0.25).cdf(upper_bound) - scipy.stats.norm(0, 0.25).cdf(lower_bound))/total_area
        if i == 2*r:
            band_ranges.append((start, 1.0))
        else:
            band_ranges.append((start, start+2*len1))
        start = start + 2*len1

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
    generate_band_ranges(args.resolution)

    dir_list = os.listdir(args.dir)
    for directory in dir_list:
        component_id = directory
        files = os.listdir(args.dir+directory)
        for file in files:
            if ".csv" in file:
                sensor_data = pd.DataFrame(pd.read_csv(args.dir + directory + '/'+ file, header=None)).to_numpy()
                data_avg = np.mean(sensor_data, axis=1)
                # print(data_avg)
                data_std = np.std(sensor_data, axis=1)
                # print(data_std)
                normalized_data = normalize(sensor_data)
                # print(normalized_data.shape)
                quantized_data = np.vectorize(getBandNumber)(normalized_data)
                # print(sensor_quantized_data)
                # get_window_avg(normalized_data, args.window, args.shift)
                get_quantized_window(quantized_data, args.window, args.shift)
                write_words(normalized_data, quantized_data, args.shift, args.window, file, component_id, data_avg, data_std)
    
    print(len(feature_dict))


def get_quantized_window(data, window, shift):
    for i, row in enumerate(data):
        h = 0
        while h + window < len(row):
            # print("h = ", h, " window values ", row[h:h+window])
            temp = [i, h, row[h:h+window].tolist()]
            h += shift
            # print(temp)
        break

def get_window_avg(data, window, shift):
    for i, row in enumerate(data):
        h = 0
        while h + window < len(row):
            # print("h = ", h, " window values ", row[h:h+window])
            val = row[h:h+window].tolist()
            temp = [i, h, mean(val)]
            h += shift
            # print(temp)
        break

def write_words(normalized_data, quantized_data, shift, window, file, component_id, data_avg, data_std):
    file_name = file.replace(".csv", "")
    data = quantized_data
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
    output_file_name = OUTPUT_FILE_PATH + "/" + file_name + "_words.csv"
    with open(output_file_name, mode='a') as output:
        fp = csv.writer(output, delimiter=',')
        while sensor < max_sensors:
            for t in range(0, max_time, shift):
                row = [component_id, file, sensor, data_avg[sensor], data_std[sensor], mean(normalized_data[sensor][t:t+window].tolist())]
                row += data[sensor][t:t + window].tolist()
                if len(row) == 6 + window:
                    # prune words which do not have enough data
                    word = [component_id, sensor]
                    word.extend(row[6:])
                    feature_dict.add(tuple(word))
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
    total_area = scipy.stats.norm(0, 0.25).cdf(1) - scipy.stats.norm(0, 0.25).cdf(-1)
    for i in range(1, 2 * r + 1):
        lower_bound = (i - r - 1) / r
        upper_bound = (i - r) / r
        gaussian_bands.append(gaussian_bands[-1] + 2 * (scipy.stats.norm(MEAN, STANDARD_DEVIATION).cdf(upper_bound) -
                                                        scipy.stats.norm(MEAN, STANDARD_DEVIATION).cdf(lower_bound))/total_area)
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
