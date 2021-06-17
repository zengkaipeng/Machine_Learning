import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
from utils import CalMu_Sigma
from tqdm import tqdm

if __name__ == '__main__':
    if not os.path.exists('Visualize'):
        os.mkdir('Visualize')

    K2beta = {}
    for k in range(1, 9):
        with open(f'result/logistic_kpart/betas_{k}.pkl', 'rb') as Fin:
            K2beta[k] = pickle.load(Fin)

    for i in range(10):
        Bs = [K2beta[x][i][-1] for x in range(1, 9)]
        Xs = list(range(1, 9))
        plt.plot(Xs, Bs, label=f'Number {i}')

    plt.xlabel('$\\frac{n_{neg}}{n_{pos}}$')
    plt.legend()
    # plt.savefig('Visualize/term_b.png')
    plt.show()

    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])

    plt.clf()
    INFO = {}
    for i in tqdm(range(10)):
        Mup, Mun, Gxp, Gxn = [], [], [], []
        for j in range(1, 9):
            beta = K2beta[j][i]
            Xpos = train_features[train_labels == i]
            Xneg = train_features[train_labels != i]
            Ppos = np.dot(Xpos, beta[:-1])
            Pneg = np.dot(Xneg, beta[:-1])
            Mpos, Gpos = CalMu_Sigma(Ppos)
            Mneg, Gneg = CalMu_Sigma(Pneg)
            Mup.append(Mpos)
            Mun.append(Mneg)
            Gxp.append(Gpos)
            Gxn.append(Gneg)
        INFO[i] = (Mup, Mun, Gxp, Gxn)

    for i in range(10):
        plt.plot(list(range(1, 9)), INFO[i][0], label=f'Number {i}')

    plt.xlabel('$\\frac{n_{neg}}{n_{pos}}$')
    plt.legend()
    plt.show()
    # plt.savefig('Visualize/Mu+.png')

    plt.clf()
    for i in range(10):
        plt.plot(list(range(1, 9)), INFO[i][1], label=f'Number {i}')

    plt.xlabel('$\\frac{n_{neg}}{n_{pos}}$')
    plt.legend()
    # plt.savefig('Visualize/Mu-.png')
    plt.show()

    plt.clf()
    for i in range(10):
        plt.plot(list(range(1, 9)), INFO[i][2], label=f'Number {i}')

    plt.xlabel('$\\frac{n_{neg}}{n_{pos}}$')
    plt.legend()
    # plt.savefig('Visualize/simga+.png')
    plt.show()

    plt.clf()
    for i in range(10):
        plt.plot(list(range(1, 9)), INFO[i][3], label=f'Number {i}')

    plt.xlabel('$\\frac{n_{neg}}{n_{pos}}$')
    plt.legend()
    # plt.savefig('Visualize/simga-.png')
    plt.show()
