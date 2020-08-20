import os
import sys
from scripts.n_gram_creator import gen_ngram_from_files, gen_ngram_from_dir
import pickle
import pandas as pd


def creator(FILE_PATHS, N_GRAM_SIZE, NUM_FILES, FREQ_FILE=None, OUTPUT_FILE=None):
    df = gen_ngram_from_files(FILE_PATHS, N_GRAM_SIZE, NUM_FILES, FREQ_FILE, OUTPUT_FILE)
    return df


def infer(MODEL_PATH, df):
    # load clf
    clf = pickle.load(open(MODEL_PATH, 'rb'))

    # load data
    X_ = df.iloc[:, :]
    y_pred = clf.predict(X_)
    print(y_pred)


if __name__ == "__main__":
    # FILE_PATHS = ['/media/tunguyen/TuTu_Passport/MTAAV/bin/none1/0b0f20a32fcf4fe6c49e1b3cb8c8ba50a19401589e7b8d3120d9d898653289a8',
    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/none1/0cba2c048182195fd91bb7b8298b47062ea7ad86772df0b98b67c3ff61679508',

    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/malware/0d1dd13fe9d14547711ca8bd80233377a0ec691473ea04298d30747bf93f093a',
    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/malware/1a888cee80582f268c8c9bd914ed196f3135161e2dd2418ac77ceacaadd41f59',

    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/benign/1a2ff8d363069b817fb2e421c16ccc43d4a4554ab9c3affa0b8550d66c002c2a',
    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/benign/1eba06ec93001087fdefd0c520ab665a75d295056ef36ed79285960252a59618']
    # df = creator(FILE_PATHS=FILE_PATHS, N_GRAM_SIZE=3, NUM_FILES=1, FREQ_FILE='/media/tunguyen/TuTu_Passport/MTAAV/ngram/benign_3gram_freq_arr')
    # # df = pd.read_csv('/media/tunguyen/TuTu_Passport/MTAAV/ngram/none1_3gram_test.csv', header=None)
    # infer(MODEL_PATH='/media/tunguyen/TuTu_Passport/MTAAV/ngram/clf_3gram', df=df)

    args = sys.argv
    INPUT_DIR = args[1]
    OUTPUT_FILE = args[2]
    NUM_FILES = int(args[3])
    N_GRAM_SIZE = int(args[4])
    gen_ngram_from_dir(INPUT_DIR, N_GRAM_SIZE, NUM_FILES, OUTPUT_FILE)
