
from random import shuffle

import numpy as np


def Random_Sample(train_features, train_labels, num=20000):
    Indexes = list(range(len(train_features)))
    shuffle(Indexes)
    features, labels = [], []
    for x in range(num):
        features.append(train_features[Indexes[x]])
        labels.append(train_labels[Indexes[x]])

    return np.array(features), np.array(labels)


def Dshuffle(features, labels, lens):
    Index = list(range(lens))
    shuffle(Index)
    return features[Index], labels[Index]
