import scripts.n_gram_analyzer as nga
import os
import sys
import numpy as np
import pandas as pd
import csv
import pickle

MAPPING = {
    'benign': 0,
    'malware': 1
}

def main(args):
    INPUT_DIR = args[1]
    OUTPUT_FILE = args[2]
    NUM_FILES = int(args[3])
    N_GRAM_SIZE = int(args[4])

    gen_ngram_from_dir(INPUT_DIR, N_GRAM_SIZE, NUM_FILES, OUTPUT_FILE)


# def gen_ngram_from_dir(INPUT_DIR, N_GRAM_SIZE, NUM_FILES, OUTPUT_FILE):
#     count = 0
#     file_paths = []
#     for filename in os.listdir(INPUT_DIR):
#         if count >= NUM_FILES:
#             break
#         count += 1
#         file_paths.append(INPUT_DIR + '/' + filename)

#     freq_arr = get_freq_arr(file_paths, N_GRAM_SIZE, NUM_FILES)

#     df = fcn_gen_ngram(file_paths, N_GRAM_SIZE, NUM_FILES, freq_arr, OUTPUT_FILE)

#     # save freq_arr
#     with open('{}/{}_freq_arr'.format(os.path.dirname(OUTPUT_FILE), OUTPUT_FILE.split('/')[-1].split('.')[0]), 'wb') as f:
#         pickle.dump(freq_arr, f)

def gen_ngram_from_dir(INPUT_DIR, N_GRAM_SIZE, NUM_FILES, OUTPUT_FILE):
    count = 0
    file_paths = []
    lbls_dict = {}
    for lbl in os.listdir(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR+'/'+lbl):
            if count >= NUM_FILES:
                break
            count += 1
            file_paths.append(INPUT_DIR+'/'+lbl+'/'+filename)
            lbls_dict[filename] = MAPPING[lbl]

    freq_arr = get_freq_arr(file_paths, N_GRAM_SIZE, NUM_FILES)

    df = fcn_gen_ngram(file_paths, N_GRAM_SIZE,
                       NUM_FILES, freq_arr, lbls_dict, OUTPUT_FILE)

    # save freq_arr
    with open('{}/{}_freq_arr'.format(os.path.dirname(OUTPUT_FILE), OUTPUT_FILE.split('/')[-1].split('.')[0]), 'wb') as f:
        pickle.dump(freq_arr, f)


def gen_ngram_from_files(FILE_PATHS, N_GRAM_SIZE, NUM_FILES, FREQ_FILE, OUTPUT_FILE=None):
    with open(FREQ_FILE, 'rb') as f:
        freq_arr = pickle.load(f)
    df = fcn_gen_ngram(FILE_PATHS, N_GRAM_SIZE,
                       NUM_FILES, freq_arr, OUTPUT_FILE)
    return df


def fcn_gen_ngram(FILE_PATHS, N_GRAM_SIZE, NUM_FILES, FREQ_AR, LABELS_DICT=None, OUTPUT_FILE=None):
    print("Highest frequencies: {}".format(FREQ_AR))

    csv_dict = dict((gram, []) for gram in FREQ_AR)
    csv_dict['label'] = []

    for filepath in FILE_PATHS:
        frequencies = nga.get_gram_frequencies(N_GRAM_SIZE, filepath, FREQ_AR)
        normalize_dict(frequencies)
        for k in csv_dict:
            if k != 'label':
                if k in frequencies:
                    csv_dict[k].append(frequencies[k])
                else:
                    csv_dict[k].append(0.0)
        csv_dict['label'].append(LABELS_DICT[filepath.split('/')[-1]])
    print('csv_dict', csv_dict)
    df = pd.DataFrame(csv_dict)
    print(df)
    if OUTPUT_FILE is not None:
        df.to_csv(OUTPUT_FILE, encoding='utf-8', index=False, header=None)
        print('Added to file: {}'.format(OUTPUT_FILE))

    return df


def get_freq_arr(filepaths, ng_size, max_f):
    frequency_dict = {}

    count = 0
    for filepath in filepaths:
        if count >= max_f:
            break
        count += 1
        frequencies = nga.get_gram_frequencies(ng_size, filepath)
        normalize_dict(frequencies)
        add_to_freq_dict(frequency_dict, frequencies)

    return get_highest_frequencies(frequency_dict, 30)


def get_highest_frequencies(d, count):
    sorted_dict = sorted(d, key=lambda k: d[k][0], reverse=True)
    return sorted_dict[:count]


def normalize_dict(dictionary):
    factor = 1.0/sum(dictionary.values())
    for k in dictionary:
        dictionary[k] = dictionary[k]*factor


def add_to_freq_dict(freq_dict, input_dict):
    for k in input_dict:
        if k in freq_dict:
            val, count = freq_dict[k]
            freq_dict[k] = (((val*count)+input_dict[k])/(count+1), count+1)
        else:
            freq_dict[k] = (input_dict[k], 1)


if __name__ == "__main__":
    main(sys.argv)
