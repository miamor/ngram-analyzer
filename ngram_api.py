import os
import sys
from scripts.n_gram_creator import gen_ngram_from_files, gen_ngram_from_dir
import pickle
import pandas as pd


class NGRAM_module:
    def __init__(self, model_path):
        self.model_path = model_path# load clf
        self.clf = pickle.load(open(self.model_path, 'rb'))

        return 

    def creator(self, file_paths, n_gram_size, num_files, freq_file=None, output_file=None):
        df = gen_ngram_from_files(file_paths, n_gram_size, num_files, freq_file, output_file)
        return df


    def infer(self, df):
        # load data
        X = df.iloc[:, :-1]
        # y = df.iloc[:, -1].tolist()
        y_pred = self.clf.predict(X)
        print(y_pred)
        return y_pred


if __name__ == "__main__":
    # file_paths = ['/media/tunguyen/TuTu_Passport/MTAAV/bin/none1/0b0f20a32fcf4fe6c49e1b3cb8c8ba50a19401589e7b8d3120d9d898653289a8',
    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/none1/0cba2c048182195fd91bb7b8298b47062ea7ad86772df0b98b67c3ff61679508',

    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/malware/0d1dd13fe9d14547711ca8bd80233377a0ec691473ea04298d30747bf93f093a',
    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/malware/1a888cee80582f268c8c9bd914ed196f3135161e2dd2418ac77ceacaadd41f59',

    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/benign/1a2ff8d363069b817fb2e421c16ccc43d4a4554ab9c3affa0b8550d66c002c2a',
    #               '/media/tunguyen/TuTu_Passport/MTAAV/bin/new_a_Dung/benign/1eba06ec93001087fdefd0c520ab665a75d295056ef36ed79285960252a59618']
    root = '/media/tunguyen/TuTu_Passport/MTAAV/bin/none/'
    file_paths = []
    total = 0
    for filename in os.listdir(root):
        file_paths.append(root+filename)
        total += 1
    df = creator(file_paths=file_paths, n_gram_size=2, num_files=total,
                 freq_file='/media/tunguyen/TuTu_Passport/MTAAV/ngram/all_2gram_freq_arr',
                 output_file='/media/tunguyen/TuTu_Passport/MTAAV/ngram/none_2gram.csv')
    # df = pd.read_csv('/media/tunguyen/TuTu_Passport/MTAAV/ngram/none_2gram_test.csv', header=None)
    infer(MODEL_PATH='/media/tunguyen/TuTu_Passport/MTAAV/ngram/clf_2gram', df=df)

    # args = sys.argv
    # INPUT_DIR = args[1]
    # output_file = args[2]
    # num_files = int(args[3])
    # n_gram_size = int(args[4])
    # gen_ngram_from_dir(INPUT_DIR, n_gram_size, num_files, output_file)
