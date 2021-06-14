from random import shuffle
from torchvision.datasets import SVHN
from skimage import feature as feat
from tqdm import tqdm
import numpy as np


def Get_Dataset():
    train_set = SVHN('../data', split='train', download=False)
    test_set = SVHN('../data', split='test', download=False)
    extra_set = SVHN('../data', split='extra', download=False)
    return train_set, test_set, extra_set


if __name__ == '__main__':
    train_set, test_set, extra_set = Get_Dataset()
    train_len = len(train_set)

    train_features, train_labels = [], []
    for i in tqdm(range(train_len)):
        img, label = train_set[i]
        if label == 0:
            img.save('test.png')
            exit()