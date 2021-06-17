import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    with open('Visualize/tsnet_mini.pkl', 'rb') as Fin:
        INFO = pickle.load(Fin)

    train_labels = np.load('mini_train_labels.npy')

    kernel = 'poly'

    with open(f'result/SVM/support_{kernel}.pkl', 'rb') as Fin:
        supports = pickle.load(Fin)

    xscatter = plt.scatter(
        INFO[supports, 0], INFO[supports, 1],
        c=train_labels[supports]
    )
    plt.legend(*xscatter.legend_elements())
    plt.show()
