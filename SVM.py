from sklearn.svm import SVC
import numpy as np
from random import shuffle
import time
from utils import Random_Sample
import pickle
import os

if __name__ == '__main__':
    train_features_u = np.load('mini_train_features.npy')
    train_labels_u = np.load('mini_train_labels.npy')
    test_labels = np.load('test_labels.npy')
    test_features = np.load('test_features.npy')

    print('load Done')

    kernel = 'poly'

    start_time = time.time()
    svm_cla = SVC(kernel=kernel, C=1, degree=2)
    svm_cla.fit(train_features_u, train_labels_u)
    print('{} Used'.format(time.time() - start_time))
    print(svm_cla.score(test_features, test_labels))

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('result/SVM'):
        os.mkdir('result/SVM')
    with open(f'result/SVM/model_{kernel}.pkl', 'wb') as Fout:
        pickle.dump(svm_cla, Fout)

    with open(f'result/SVM/support_{kernel}.pkl', 'wb') as Fout:
        pickle.dump(svm_cla.support_, Fout)
