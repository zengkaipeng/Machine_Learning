import numpy as np
import os
import pickle
import json

if __name__ == '__main__':
    train_features = np.load('mini_train_features.npy')
    train_labels = np.load('mini_train_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')
    