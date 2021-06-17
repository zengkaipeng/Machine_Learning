import numpy as np
import os
import pickle
from tqdm import tqdm
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


def Kernel_Matric(
    Xis, ker='rbf', from_file=False, fdir='',
    d=2, sigma=1, Norms=None
):
    if from_file and os.path.exists(fdir):
        kernel = np.load(fdir)
        return kernel
    else:
        if not os.path.exists('kernels'):
            os.mkdir('kernels')

        if ker == 'rbf':
            Dx = [Rbf(x, Xis, simga=sigma) for x in tqdm(Xis)]
            Dx = np.array(Dx)
            np.save('kernels/rbf.npy', Dx)
            return Dx
        if ker == 'cos':
            if Norms is None:
                Norms = (Xis * Xis).sum(axis=1).flatten()
            Dx = [Cons(x, Xis, Norms) for x in tqdm(Xis)]
            Dx = np.array(Dx)
            np.save('kernels/cos.npy', Dx)
            return Dx
        if ker == 'poly':
            Dx = [Ploy(x, Xis, d) for x in tqdm(Xis)]
            Dx = np.array(Dx)
            np.save('kernels/poly.npy', Dx)
            return Dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Grad(K, c, y):
    return np.dot(K.T, sigmoid(np.dot(K, c)) - y) + 2e-3 * np.dot(K, c)


def Loss(K, c, y):
    return np.sum(np.log(1 + np.exp(np.dot(K, c)))) -\
        np.sum(y * np.dot(K, c)) + 1e-3 * np.dot(c, np.dot(K, c))


def train(
    K, train_labels, verbose=True, step=False,
    lr=1e-1, gam1=0.9, gam2=0.999, epoch=500
):
    c = np.zeros(len(K))
    vt = np.zeros_like(c)
    Gt = np.zeros_like(c)
    for ep in range(epoch):
        grad = Grad(K, c, train_labels)

        vt = gam1 * vt + (1 - gam1) * grad
        Gt = gam2 * Gt + (1 - gam2) * (grad * grad)
        vth = vt / (1 - gam1 ** (ep + 1))
        Gth = Gt / (1 - gam2 ** (ep + 1))
        c -= lr * (vth / np.sqrt(Gth + 1e-8))

        if verbose and (ep + 1) % 50 == 0:
            print('Epoch = {} Loss = {}'.format(
                ep + 1, Loss(K, c, train_labels)
            ))

        if step != 0 and (ep + 1) % step == 0:
            lr /= 10
    return c


def GetKlines(
    data, train_data, ker='poly',
    sigma=1, d=2, Norms=None, verbose=True
):
    lx = data if not verbose else tqdm(data)
    Answer = []
    for x in lx:
        if ker == 'rbf':
            kline = Rbf(x, train_data, simga=sigma)
        if ker == 'poly':
            kline = Ploy(x, train_data, d)
        if ker == 'cos':
            kline = Cons(x, train_data, Norms)
        Answer.append(kline)
    return np.array(Answer)


if __name__ == '__main__':
    train_features = np.load('mini_train_features.npy')
    train_labels = np.load('mini_train_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    kern = 'cos'
    K = Kernel_Matric(
        train_features, ker=kern,
        from_file=True, fdir=f'kernels/{kern}.npy'
    )
    print(K.shape)
    print(K.max(), K.min())

    Norms = np.sqrt((train_features * train_features).sum(axis=1)).flatten()
    print(Norms.max(), Norms.min())

    Lab2C = {}
    for i in range(10):
        print(f"[INFO] {i}")
        newlabels = np.zeros_like(train_labels)
        newlabels[train_labels == i] = 1
        C = train(K, newlabels, epoch=500, step=0)
        Lab2C[i] = C

    Klines = GetKlines(test_features, train_features, ker=kern, Norms=Norms)
    As = []
    for i in range(10):
        As.append(sigmoid(np.dot(Klines, Lab2C[i])))

    As = np.array(As)
    Predict = As.argmax(axis=0)
    print(Predict.shape)
    Corr = np.sum(Predict == test_labels)
    print(Corr / len(test_labels))

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('result/kernel_logistic'):
        os.mkdir('result/kernel_logistic')

    with open(f'result/kernel_logistic/Cs_{kern}.pkl', 'wb') as Fout:
        pickle.dump(Lab2C, Fout)
