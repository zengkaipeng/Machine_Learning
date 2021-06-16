import numpy as np
import os
import pickle
import json


def Rbf(Xi, Xjs, simga):
    Gap = Xjs - Xi
    dists = (Gap * Gap).sum(axis=1).flatten()
    Line = np.exp(-dists / (2 * simga ** 2))
    return Line


def Cons(Xi, Xjs, Norms):
    Xnorm = Xi.dot(Xi)
    return np.dot(Xjs, Xi) / Norms / Xnorm


def Ploy(Xi, Xjs, d):
    return np.dot(Xjs, Xi) ** d


def Kernel_Matric(Xis, ker='rbf', from_file=False, fdir=''):
    if from_file:
        kernel = np.load(fdir)
        return kernel
    else:
        if not os.path.exists('kernels'):
            os.mkdir('kernels')

        if ker == 'rbf':
            sigma = 0.1
            Dx = [Rbf(x, Xis, simga) for x in Xis]
            Dx = np.array(Dx)
            np.save(Dx, 'kernels/rbf.npy')
            return Dx
        if ker == 'cos':
            Norms = (Xis * Xis).sum(axis=1).flatten()
            Dx = [Cons(x, Xis) for x in Xis]
            Dx = np.array(Dx)
            np.save(Dx, 'kernels/cos.npy')
            return Dx
        if ker == 'poly':
            d = 3
            Dx = [Ploy(x, Xis, d) for x in Xis]
            Dx = np.array(Dx)
            np.save(Dx, 'kernels/poly.npy')
            return Dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Loss(X, c, y):
    return np.sum(
        -y * np.log(sigmoid(np.dot(X, beta))) -
        (1 - y) * np.log(1 - sigmoid(np.dot(X, beta)))
    )




if __name__ == '__main__':
    train_features = np.load('mini_train_features.npy')
    train_labels = np.load('mini_train_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    K = Kernel_Matric(train_features, ker='poly')

