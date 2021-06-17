import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    with open('Visualize/tsnet_mini.pkl', 'rb') as Fin:
        INFO = pickle.load(Fin)

    INFO = np.array(INFO)

    train_labels = np.load('mini_train_labels.npy')
    with open('result/kernel_logistic/Cs_poly.pkl', 'rb') as Fin:
        Cs = pickle.load(Fin)

    Sels = []
    Cols = []
    for i in range(10):
        Ax = np.fabs(Cs[i])
        top10 = np.argsort(-Ax)[:50]
        Sels.append(top10)
        print(train_labels[top10])
        Cols += [i] * 50

    Sels = np.concatenate(Sels)

    xscatter = plt.scatter(INFO[Sels, 0], INFO[Sels, 1], c=Cols)
    plt.legend(*xscatter.legend_elements())

    plt.show()