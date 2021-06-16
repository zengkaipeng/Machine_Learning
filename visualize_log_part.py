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
    Kpart = 8

    if Kpart < 9:
        with open(f'result/logistic_kpart/betas_{Kpart}.pkl', 'rb') as Fin:
            Log_Beta = pickle.load(Fin)
    else:
        with open('result/logistic/betas.pkl', 'rb') as Fin:
            Log_Beta = pickle.load(Fin)

    if not os.path.exists('Visualize'):
        os.mkdir('Visualize')

    train_features = np.hstack([train_features, np.ones((train_len, 1))])

    Cross_Point = {}

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

        MuPos, GxPos = CalMu_Sigma(Projectpos)
        MuNeg, GxNeg = CalMu_Sigma(Projectneg)
        Xs = np.arange(-16, 16, 0.001)
        Fun1 = Gauss(Xs, MuPos, GxPos)
        Fun2 = Gauss(Xs, MuNeg, GxNeg)
        plt.plot(Xs, Fun1, label='Positive')
        plt.plot(Xs, Fun2, label='Negative')
        plt.xlabel('$\\widetilde{{X}}^T\\beta_{}$'.format(i))
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'Visualize/logistic_kpart_{Kpart}_{i}.png')
        Ans, Agp = None, 0
        for ip, x in enumerate(Xs):
            gap = abs(Fun2[ip] - Fun1[ip])
            if x < min(MuPos, MuNeg) or x > max(MuPos, MuNeg):
                continue
            if Ans is None or gap < Agp:
                Agp, Ans = gap, x
        Cross_Point[i] = Ans

    with open(
        'result/logistic_kpart/Cross_Point_{}.json'.format(Kpart), 'w'
    ) as Fout:
        json.dump(Cross_Point, Fout, indent=4)
