import pickle
import os
import numpy as np
from utils import Dshuffle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Loss(X, beta, y):
    return np.sum(
        -y * np.log(sigmoid(np.dot(X, beta))) -
        (1 - y) * np.log(1 - sigmoid(np.dot(X, beta)))
    )


def Grad(X, beta, y, Lambda):
    Lasso = np.zeros_like(beta)
    Lasso[beta > 0] = 1
    Lasso[beta < 0] = -1
    return np.dot(X.T, sigmoid(np.dot(X, beta)) - y) + Lambda * Lasso


def train(
    train_features, train_labels, batch_size=256,
    lr=1e-2, gam1=0.9, gam2=0.999, epoch=500,
    Lambda=1e-2, verbose=True
):
    beta = np.zeros(len(train_features[0]))
    vt = np.zeros(len(train_features[0]))
    Gt = np.zeros(len(train_features[0]))
    iLam = Lambda
    for ep in range(epoch):
        loss = 0
        for Idx in range(0, len(train_features), batch_size):
            Subx = train_features[Idx: Idx + batch_size]
            Y = train_labels[Idx: Idx + batch_size]
            grad = Grad(Subx, beta, Y, iLam)

            vt = gam1 * vt + (1 - gam1) * grad
            Gt = gam2 * Gt + (1 - gam2) * (grad * grad)
            vth = vt / (1 - gam1 ** (ep + 1))
            Gth = Gt / (1 - gam2 ** (ep + 1))
            beta -= lr * (vth / np.sqrt(Gth + 1e-8))

            # beta -= lr * grad
            loss += Loss(Subx, beta, Y)

        if verbose and (ep + 1) % 50 == 0:
            print('Epoch = {} Loss = {}'.format(ep + 1, loss))

        if (ep + 1) % 100 == 0:
            iLam /= 10
        train_features, train_labels = Dshuffle(
            train_features, train_labels, trainlen
        )

    return beta


if __name__ == '__main__':
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    extra_features = np.load('extra_features.npy')
    extra_labels = np.load('extra_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = np.concatenate([train_features, extra_features])
    train_labels = np.concatenate([train_labels, extra_labels])

    batch_size = 256
    train_len = len(train_features)

    Lab2beta = {}

    for i in range(10):
        new_labels = np.zeros(len(train_features))
        new_labels[train_labels == i] = 1
        new_features = np.hstack([train_features, np.ones((train_len, 1))])
        print(np.sum(new_labels))
        beta = train(new_features, new_labels)
        Lab2beta[i] = beta

    Answer, Labs = [], []

    for k, v in Lab2beta.items():
        testf = np.hstack([test_features, np.ones((len(test_features), 1))])
        Answer.append(sigmoid(np.dot(testf, v)))
        Labs.append(k)

    Answer = np.array(Answer)
    Predict = Answer.argmax(axis=0)
    Predict = np.array([Labs[x] for x in Predict])
    Corr = np.sum(Predict == test_labels)
    print(Corr / len(test_features))

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('result/logistic_lasso'):
        os.mkdir('result/logistic_lasso')

    with open('result/logistic_lasso/betas.pkl', 'wb') as Fout:
        pickle.dump(Lab2beta, Fout)
