from sklearn.svm import SVC
import numpy as np
from random import shuffle
import time
from utils import Random_Sample
import pickle
import os

if __name__ == '__main__':
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])

    print('load Done')

    kernel = 'rbf'

    start_time = time.time()
    svm_cla = SVC(kernel=kernel, C=1, degree=2)
    svm_cla.fit(train_features, train_labels)
    print('{} Used'.format(time.time() - start_time))
    print(svm_cla.score(test_features, test_labels))

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('result/SVM'):
        os.mkdir('result/SVM')
    with open(f'result/SVM/model_{kernel}_Full.pkl', 'wb') as Fout:
        pickle.dump(svm_cla, Fout)
