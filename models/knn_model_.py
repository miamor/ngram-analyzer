import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def main(args):
    try:
        START_N_GRAM_SIZE = int(args[1])
        MAX_NGRAM = int(args[2])
        CSV_DIRECTORY_PATH = args[3]
        STARTING_NEIGHBOR = int(args[4])
        END_NEIGHBOR = int(args[5])
    except Exception as e:
        print(e)
        print('Expected: python knn_model.py <starting n-gram size> <max n-gram size> <path to csv\'s> <starting neighbor count> <ending neighbor count>')
        sys.exit()

    for i in range(START_N_GRAM_SIZE, MAX_NGRAM+1):
        benign_file = ''
        malware_file = ''
        try:
            benign_file = CSV_DIRECTORY_PATH + '/benign_' + str(i) + 'gram.csv'
            malware_file = CSV_DIRECTORY_PATH + '/malware_' + str(i) + 'gram.csv'
        except Exception as e:
            print(e)
            sys.exit()

        benign_df = pd.read_csv(benign_file, header=None)
        malware_df = pd.read_csv(malware_file, header=None)

        X = benign_df.iloc[:, :]
        y = [0] * X.shape[0]
        X2 = malware_df.iloc[:, :]
        y2 = [1] * X2.shape[0]

        X = X.append(X2).values
        y += y2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=0)

        for n in range(STARTING_NEIGHBOR, END_NEIGHBOR+1):
            clf = KNeighborsClassifier(n_neighbors=n)
            clf.fit(X_train, y_train)
            print('Score with {}-gram and {}-neigbors: {}'.format(i,
                                                                  n, clf.score(X_test, y_test)))


if __name__ == "__main__":
    main(sys.argv)
