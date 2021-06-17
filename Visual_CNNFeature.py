import os
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from sklearn.manifold import TSNE

if __name__ == '__main__':
	totfeatures = torch.load('tot_test_features.tov').numpy()
	test_labels = torch.load('tot_test_labels.tov').numpy()

	TSNER = TSNE(n_components=2)
	new_fea = TSNER.fit_transform(totfeatures)

	xscatter = plt.scatter(new_fea[:, 0], new_fea[:, 1], c=test_labels)
	plt.legend(*xscatter.legend_elements())
	# plt.savefig('Visualize/CNN_Features.png')
	plt.show()