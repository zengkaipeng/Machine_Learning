import json
import math
import numpy as np
from tqdm import tqdm
import os
import pickle


def Binary_LDA(features, labels, verbose=True):
    totlen = len(features)
    Xin = np.array([features[i] for i in range(totlen) if labels[i] == 1])
    Xout = np.array([features[i] for i in range(totlen) if labels[i] != 1])
    muplus = np.mean(Xin, axis=0)
    muneg = np.mean(Xout, axis=0)

    if verbose:
        print('[INFO] MEANS DONE')

    Sigmain = np.matmul((Xin - muplus).T, Xin - muplus)
    Sigmaout = np.matmul((Xout - muneg).T, Xout - muneg)

    if verbose:
        print('[INFO] SIGMA DONE')

    SW = Sigmaout + Sigmain
    SWinv = np.linalg.inv(SW)
    beta = np.dot(SWinv, (muplus - muneg))
    return beta


def CalMu_Sigma(Array):
    Mu = np.mean(Array)
    Gx = np.mean((Array - Mu) ** 2)
    return Mu, Gx


def Gauss(x, Mu, Gx):
    return 1 / (np.sqrt(2 * math.pi * Gx)) * \
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

    Lab2beta = {}
    for x in range(10):
        new_labels = np.zeros(len(train_features))
        new_labels[train_labels == x] = 1
        beta = Binary_LDA(train_features, new_labels)
        Lab2beta[x] = beta

    Lab2Info = {}
    for x in range(10):
        Xpos = train_features[train_labels == x]
        Xneg = train_features[train_labels != x]
        Projection_pos = np.dot(Xpos, Lab2beta[x])
        Projection_neg = np.dot(Xneg, Lab2beta[x])
        Mupos, Gxpos = CalMu_Sigma(Projection_pos)
        Muneg, Gxneg = CalMu_Sigma(Projection_neg)
        Lab2Info[x] = (Mupos, Gxpos, Muneg, Gxneg)

    Probs, Labs = [], []
    train_len = len(train_features)
    for x in range(10):
        Projection = np.dot(test_features, Lab2beta[x])
        Mupos, Gxpos, Muneg, Gxneg = Lab2Info[x]
        ProbPos = Gauss(Projection, Mupos, Gxpos)
        ProbNeg = Gauss(Projection, Muneg, Gxneg)
        Py1 = np.sum(train_labels == x) / train_len
        Prob = (ProbPos) / (ProbPos * Py1 + ProbNeg * (1 - Py1))

        Probs.append(Prob)
        Labs.append(x)

    Probs = np.array(Probs)
    Predict = Probs.argmax(axis=0)
    Predict = np.array([Labs[x] for x in Predict])

    Corr = np.sum(Predict == test_labels)
    print(Corr / len(test_features))

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('result/LDA'):
        os.mkdir('result/LDA')

    with open('result/LDA/betas.pkl', 'wb') as Fout:
        pickle.dump(Lab2beta, Fout)

    with open('result/LDA/Gauss_Para.json', 'w') as Fout:
        json.dump(Lab2Info, Fout, indent=4)
