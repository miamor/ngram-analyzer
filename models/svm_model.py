import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle

def main(args):
    try:
        START_N_GRAM_SIZE = int(args[1])
        MAX_NGRAM = int(args[2])
        CSV_DIRECTORY_PATH = args[3]
        OUTPUT_DIR = args[4]

    except Exception as e:
        print(e)
        print('Expected: python svm_model.py <starting n-gram size> <max n-gram size> <path to csv\'s> <output dir>')
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
        print('X', X)
        # print('y', y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        pickle.dump(clf, open(OUTPUT_DIR+'/clf_' + str(i) + 'gram', 'wb'))
        print('\n* Score with {}-gram: {}'.format(i, clf.score(X_test, y_test)))

        # y_pred = clf.predict(X_test)
        # fir_matrix = confusion_matrix(y_test, y_pred)
        # print(fir_matrix)

        # Plot non-normalized confusion matrix
        titles_options = [("ngram={} | Confusion matrix, without normalization".format(i), None),
                        ("ngram={} | Normalized confusion matrix".format(i), 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(clf, X, y,
                                        display_labels=[0,1],
                                        # cmap=plt.cm.Blues,
                                        normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

    plt.show()


    # Infere
    infer_df = pd.read_csv('/media/tunguyen/TuTu_Passport/MTAAV/ngram/benign_3gram_1.csv', header=None)

    X_ = infer_df.iloc[:, :]
    y_ = [0] * X_.shape[0]
    y_pred = clf.predict(X_)
    print(y_pred, y_)



if __name__ == "__main__":
    main(sys.argv)
