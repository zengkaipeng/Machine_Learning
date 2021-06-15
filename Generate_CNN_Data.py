import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from torchvision import transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

Norm = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

Transfor = transforms.Compose([
    transforms.ToTensor(),
    Norm
])


def Get_Dataset():
    train_set = SVHN(
        '../data', split='train',
        transform=Transfor, download=False
    )
    test_set = SVHN(
        '../data', split='test',
        download=False, transform=Transfor
    )
    extra_set = SVHN(
        '../data', split='extra',
        download=False, transform=Transfor
    )
    return train_set, test_set, extra_set


if __name__ == '__main__':
    train_set, test_set, extra_set = Get_Dataset()

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=True)
    extra_loader = DataLoader(extra_set, batch_size=256, shuffle=True)

    test_features, test_labels = [], []
    for Idx, (data, label) in enumerate(tqdm(test_loader)):
        test_features.append(data)
        test_labels.append(label)

    test_features = torch.cat(test_features, 0)
    test_labels = torch.cat(test_labels, 0)

    print(test_features.shape)
    print(test_labels.shape)
    torch.save(test_features, "CNN_Data/test_features.tov")
    torch.save(test_labels, 'CNN_Data/test_labels.tov')

    train_features, train_labels = [], []
    for Idx, (data, label) in enumerate(tqdm(train_loader)):
        train_features.append(data)
        train_labels.append(label)

    for Idx, (data, label) in enumerate(tqdm(extra_loader)):
        train_features.append(data)
        train_labels.append(label)
        if Idx == 200:
            break

    train_features = torch.cat(train_features, 0)
    train_labels = torch.cat(train_labels, 0)
    print(train_features.shape, train_labels.shape)
    torch.save(train_features, 'CNN_Data/train_features.tov')
    torch.save(train_labels, 'CNN_Data/train_labels.tov')
