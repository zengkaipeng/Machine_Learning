import json
import math
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Gauss(x, Mu, Gx):
    return 1 / (np.sqrt(2 * math.pi) * np.sqrt(Gx)) * \
        np.exp(-(x - Mu) * (x - Mu) / (2 * Gx))


def CalMu_Sigma(Array):
    Mu = np.mean(Array)
    Gx = np.mean((Array - Mu) ** 2)
    return Mu, Gx


if __name__ == '__main__':
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])
    train_len = len(train_features)
    Kpart = 1
    
    with open(f'result/logistic_kpart/betas_{Kpart}.pkl', 'rb') as Fin:
        Log_Beta = pickle.load(Fin)
    """
    with open('result/logistic/betas.pkl', 'rb') as Fin:
        Log_Beta = pickle.load(Fin)
    """
    
    if not os.path.exists('Visualize'):
        os.mkdir('Visualize')

    train_features = np.hstack([train_features, np.ones((train_len, 1))])

    for i in range(10):
        Xpos = train_features[train_labels == i]
        Xneg = train_features[train_labels != i]

        Projectpos = np.dot(Xpos, Log_Beta[i])
        Projectneg = np.dot(Xneg, Log_Beta[i])
        plt.clf()
        plt.hist(
            Projectpos, bins=200, label='Positive Samples',
            density=True, alpha=0.5
        )

        plt.hist(
            Projectneg, bins=200, label='Negative Samples',
            density=True, alpha=0.5
        )
        plt.legend()
        plt.show()
        exit()
