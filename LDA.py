import numpy as np


def Binary_LDA(features, labels):
    dim = len(features[0])
    sumin = np.zeros(dim)
    sumout = np.zeros(dim)
    Sigmain = np.zeros((dim, dim))
    Sigmaout = np.zeros((dim, dim))
    within, outside = 0, 0
    for i, fea in enumerate(features):
        if labels[i] == 1:
            sumin += fea
            within += 1
        else:
            sumout += fea
            outside += 1

    muplus = sumin / within
    muneg = sumout / outside

    mugap = muplus - muneg
    SB = np.matmul(mugap.reshape(-1, 1), mugap)
    for i, fea in enumerate(features):
        if labels[i] == 1:
            vecgap = fea - muplus
            Sigmain += np.matmul(vecgap.reshape(-1, 1), vecgap)
        else:
            vecgap = fea - muneg
            Sigmaout += np.matmul(vecgap.reshape(-1, 1), vecgap)

    """
    Sigmain = Sigmain / within
    Sigmaout = Sigmaout / outside
    """
    SW = Sigmaout + Sigmain
    SW_inv = np.linalg.inv(SW)
    SWSB = np.matmul(SW_inv, SB)
    engval, engvec = np.linalg.eig(SWSB)
    pos = np.argmax(engval)
    beta = engvec[:, pos]
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

    Lab2beta = {}
    for x in range(10):
        new_labels = np.zeros(len(train_features[0]))
        new_labels[train_labels == x] = 1
        beta = Binary_LDA(train_features, new_labels)
        