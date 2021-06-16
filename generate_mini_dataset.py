import pickle
import os
import numpy as np
from random import shuffle
import pickle

if __name__ == '__main__':
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])

    Indlist = list(range(len(train_features)))
    shuffle(Indlist)

    with open('mini_selected.pkl', 'wb') as Fout:
        pickle.dump(Indlist[:10000], Fout)

    np.save('mini_train_features.npy', train_features[Indlist[:10000]])
    np.save('mini_train_labels.npy', train_labels[Indlist[:10000]])
