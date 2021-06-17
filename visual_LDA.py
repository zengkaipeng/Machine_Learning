import json
import math
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

from utils import CalMu_Sigma


def Gauss(x, Mu, Gx):
    return 1 / (np.sqrt(2 * math.pi) * np.sqrt(Gx)) * \
        np.exp(-(x - Mu) * (x - Mu) / (2 * Gx))


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

    with open('result/LDA/betas.pkl', 'rb') as Fin:
        INFO = pickle.load(Fin)

    for i in range(10):
        Xpos = train_features[train_labels == i]
        Xneg = train_features[train_labels != i]

        Projectpos = np.dot(Xpos, INFO[i])
        Projectneg = np.dot(Xneg, INFO[i])
        plt.clf()
        plt.hist(
            Projectpos, bins=200, label='Positive Samples',
            density=True, alpha=0.5
        )

        plt.hist(
            Projectneg, bins=200, label='Negative Samples',
            density=True, alpha=0.5
        )

        MuPos, GxPos = CalMu_Sigma(Projectpos)
        MuNeg, GxNeg = CalMu_Sigma(Projectneg)
        
        Xmin = min(Projectneg.min(), Projectpos.min())
        Xmax = max(Projectneg.max(), Projectpos.max())
        Xs = np.arange(Xmin, Xmax, 0.000001)
        Fun1 = Gauss(Xs, MuPos, GxPos)
        Fun2 = Gauss(Xs, MuNeg, GxNeg)
        plt.plot(Xs, Fun1, label='Positive')
        plt.plot(Xs, Fun2, label='Negative')
        
        plt.xlabel('$\\widetilde{{X}}^T\\beta_{}$'.format(i))
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('Visualize/LDA_{}.png'.format(i))
