from sklearn.svm import SVC
import numpy as np
from random import shuffle
import time
from utils import Random_Sample


if __name__ == '__main__':
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])

    train_features_u, train_labels_u = Random_Sample(
        train_features, train_labels, num=20000
    )

    print('load Done')

    start_time = time.time()
    svm_cla = SVC(kernel='rbf', C=1)
    svm_cla.fit(train_features_u, train_labels_u)
    print('{} Used'.format(time.time() - start_time))
    print(svm_cla.score(test_features, test_labels))
