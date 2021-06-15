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


if __name__ == '__main__':
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])

    with open('result/LDA/betas.pkl', 'rb') as Fin:
        LDA_Beta = pickle.load(Fin)

    with open('result/logistic/betas.pkl', 'rb') as Fin:
        Log_Beta = pickle.load(Fin)

    with open('result/LDA/Gauss_Para.json') as Fin:
    	LDA_Gau = json.load(Fin)

    if not os.path.exists('Visualize'):
        os.mkdir('Visualize')


    for i in range(10):
        data = train_features[train_labels == i]
        proj1 = np.dot(data, LDA_Beta[i])
        proj2 = np.dot(data, Log_Beta[i])
        prob2 = sigmoid(proj2)
        Mu, Gx = LDA_Gau[str(i)]
        prob1 = Gauss(proj1, Mu, Gx)
        print(1 / (np.sqrt(2 * math.pi) * np.sqrt(Gx)))
        Xs = list(range(len(data)))

        plt.clf()
        plt.plot(Xs, sorted(prob2.tolist()), label='logistic')
        plt.plot(Xs, sorted(prob1.tolist()), label='LDA')
        plt.legend()
        plt.savefig('Visualize/LDA_LOG_PROB_{}.png'.format(i))
