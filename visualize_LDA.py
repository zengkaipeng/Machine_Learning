import numpy as np
import pickle
from matplotlib import pyplot as plt

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
        Lab2beta = pickle.load(Fin)
    train_len = len(train_features)
    
    Datas = {}
    for x in range(train_len):
    	if train_labels[x] not in Datas:
    		Datas[train_labels[x]] = []
    	Datas[train_labels[x]].append(train_features[x])

    for k, v in Datas.items():
    	Pdata = np.dot(np.array(v), Lab2beta[k])
    	plt.hist(Pdata, bins=100, label='{}'.format(k), alpha=0.5)

    plt.legend()
    plt.show()
