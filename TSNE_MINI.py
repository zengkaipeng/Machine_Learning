import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pickle


if __name__ == '__main__':
    """
    train_feature = np.load('mini_train_features.npy')
    train_labels = np.load('mini_train_labels.npy')
    Ter = TSNE(n_components=2)
    newfea = Ter.fit_transform(train_feature)

        for i, (x, y) in enumerate(newfea):
                plt.text(
                        x, y, str(train_labels[i]),
                        color=plt.cm.Set1(train_labels[i] / 10),
                        fontdict={'weight': 'bold', 'size': 9}
                )

    with open('Visualize/tsnet_mini.pkl', 'wb') as Fout:
        pickle.dump(newfea, Fout)
    """

    with open('Visualize/tsnet_mini.pkl', 'rb') as Fin:
        newfea = pickle.load(Fin)
    train_labels = np.load('mini_train_labels.npy')
    newfea = np.array(newfea)
    xscatter = plt.scatter(newfea[:, 0], newfea[:, 1], c=train_labels)
    plt.legend(*xscatter.legend_elements())
    plt.show()
