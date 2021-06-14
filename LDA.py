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
        Projection = np.dot(Xpos, Lab2beta[x])
        Mu, Gx = CalMu_Sigma(Projection)
        Lab2Info[x] = (Mu, Gx)

    Probs, Labs = [], []
    for x in range(10):
        Projection = np.dot(test_features, Lab2beta[x])
        Mu, Gx = Lab2Info[x]
        sigma = np.sqrt(Gx)
        Prob = 1 / (np.sqrt(2 * math.pi) * sigma) * np.exp(
            - (Projection - Mu) * (Projection - Mu) / (2 * Gx)
        )
        Probs.append(Prob)
        

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('result/LDA'):
        os.mkdir('result/LDA')

    with open('result/LDA/betas.pkl', 'wb') as Fout:
        pickle.dump(Lab2beta, Fout)
