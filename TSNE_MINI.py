import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

if __name__ == '__main__':
	train_feature = np.load('mini_train_features.npy')
	train_labels = np.load('mini_train_labels.npy')
	Ter = TSNE(n_components=2)
	newfea = Ter.fit_transform(train_feature)
	plt.plot(newfea, color=train_labels)
	plt.show()
