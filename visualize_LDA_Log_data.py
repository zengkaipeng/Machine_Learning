import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

def Normed(vector):
    nr = np.sqrt(vector.dot(vector))
    return vector / nr


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

    if not os.path.exists('Visualize'):
        os.mkdir('Visualize')

    for i in range(10):
        data = train_features[train_labels == i]
        proj1 = np.dot(data, Normed(LDA_Beta[i]))
        proj2 = np.dot(data, Normed(Log_Beta[i]))

        plt.clf()
        plt.hist(proj1, bins=100, label='Projection By LDA', alpha=0.5)
        plt.hist(proj2, bins=100, label='Projection By Logistic', alpha=0.5)
        plt.legend()
        plt.savefig('Visualize/LDA_LOG_data{}.png'.format(i))




    
