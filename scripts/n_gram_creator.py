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
    input_dir = args[1]
    output_file = args[2]
    num_files = int(args[3])
    n_gram_size = int(args[4])

    gen_ngram_from_dir(input_dir, n_gram_size, num_files, output_file)


# def gen_ngram_from_dir(input_dir, n_gram_size, num_files, output_file):
#     count = 0
#     file_paths = []
#     for filename in os.listdir(input_dir):
#         if count >= num_files:
#             break
#         count += 1
#         file_paths.append(input_dir + '/' + filename)

#     freq_arr = get_freq_arr(file_paths, n_gram_size, num_files)

#     df = fcn_gen_ngram(file_paths, n_gram_size, num_files, freq_arr, output_file)

#     # save freq_arr
#     with open('{}/{}_freq_arr'.format(os.path.dirname(output_file), output_file.split('/')[-1].split('.')[0]), 'wb') as f:
#         pickle.dump(freq_arr, f)

def gen_ngram_from_dir(input_dir, n_gram_size, num_files, output_file):
    count = 0
    file_paths = []
    lbls_dict = {}
    freq_arr = []
    for lbl in sorted(os.listdir(input_dir)):
        freq_arr_tmp = []
        for filename in os.listdir(input_dir+'/'+lbl):
            if count >= num_files:
                break
            count += 1
            file_paths.append(input_dir+'/'+lbl+'/'+filename)
            lbls_dict[filename] = MAPPING[lbl]

        freq_arr_tmp = get_freq_arr(file_paths, n_gram_size, num_files)
        # freq_arr_tmp = [lbl+'_'+str(val) for val in freq_arr_tmp]

        freq_arr.extend(freq_arr_tmp)
    
    print('>> freq_arr', freq_arr)
    # save freq_arr
    with open('{}/{}_freq_arr'.format(os.path.dirname(output_file), output_file.split('/')[-1].split('.')[0]), 'wb') as f:
        pickle.dump(freq_arr, f)

    df = fcn_gen_ngram(file_paths, n_gram_size, num_files, freq_arr, lbls_dict, output_file)


def gen_ngram_from_files(file_paths, n_gram_size, num_files, freq_file, output_file=None):
    with open(freq_file, 'rb') as f:
        freq_arr = pickle.load(f)
    df = fcn_gen_ngram(file_paths, n_gram_size,
                       num_files, freq_arr=freq_arr, output_file=output_file)
    return df


def fcn_gen_ngram(file_paths, n_gram_size, num_files, freq_arr, lbls_dict=None, output_file=None):
    print("Highest frequencies: {}".format(freq_arr))

    csv_dict = dict()
    for i, gram in enumerate(freq_arr):
        # csv_dict[gram] = []
        if i < 30: lbl = 'benign'
        else: lbl = 'malware'
        csv_dict['{}_{}'.format(lbl, str(gram))] = []
    # csv_dict = dict((gram, []) for gram in FREQ_AR_DICT)
    csv_dict['label'] = []

    for filepath in file_paths:
        # print('** filepath', filepath)
        frequencies = nga.get_gram_frequencies(n_gram_size, filepath, freq_arr)
        normalize_dict(frequencies)
        # print('frequencies', frequencies)
        for i, gram in enumerate(freq_arr):
            if i < 30: lbl = 'benign'
            else: lbl = 'malware'
            k = '{}_{}'.format(lbl, str(gram))
            if gram in frequencies:
                csv_dict[k].append(frequencies[gram])
            else:
                csv_dict[k].append(0.0)

            # for lbl in ['benign', 'malware']:
            #     k = '{}_{}'.format(lbl, str(gram))
            #     if k in csv_dict:
            #         # print('\t gram=', k)
            #         if gram in frequencies:
            #             csv_dict[k].append(frequencies[gram])
            #         else:
            #             csv_dict[k].append(0.0)
        if lbls_dict is not None:
            csv_dict['label'].append(lbls_dict[filepath.split('/')[-1]])
        else: 
            csv_dict['label'].append(-1)
    print('csv_dict', csv_dict)
    df = pd.DataFrame(csv_dict)
    print(df)
    if output_file is not None:
        df.to_csv(output_file, encoding='utf-8', index=False, header=None)
        print('Added to file: {}'.format(output_file))

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
