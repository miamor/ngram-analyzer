import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

def main(args):
    try:
        START_N_GRAM_SIZE = int(args[1])
        MAX_NGRAM = int(args[2])
        CSV_DIRECTORY_PATH = args[3]
        STARTING_NEIGHBOR = int(args[4])
        END_NEIGHBOR = int(args[5])
        OUTPUT_DIR = args[6]
    except Exception as e:
        print(e)
        print('Expected: python knn_model.py <starting n-gram size> <max n-gram size> <path to csv\'s> <starting neighbor count> <ending neighbor count> <output dir>')
        sys.exit()

    for i in range(START_N_GRAM_SIZE, MAX_NGRAM+1):
        all_file = ''
        try:
            all_file = CSV_DIRECTORY_PATH + '/all_' + str(i) + 'gram.csv'
        except Exception as e:
            print(e)
            sys.exit()

        all_df = pd.read_csv(all_file, header=None)

        X = all_df.iloc[:, :-1]
        y = all_df.iloc[:, -1].tolist()
        print('X', X.shape)
        # print('X', X)
        # print('y', y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

        for n in range(STARTING_NEIGHBOR, END_NEIGHBOR+1):
            clf = KNeighborsClassifier(n_neighbors=n)
            clf.fit(X_train, y_train)
            pickle.dump(clf, open(OUTPUT_DIR+'/clf_knn_'+str(i)+'gram_n='+str(n), 'wb'))
            print('Score with {}-gram and {}-neigbors: {}'.format(i, n, clf.score(X_test, y_test)))


if __name__ == "__main__":
    main(sys.argv)
