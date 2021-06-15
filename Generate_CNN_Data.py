import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from torchvision import transforms
from torchvision.datasets import SVHM
from torch.utils.data import DataLoader

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
    train_set, test_set, extra_set = Get_Dataset

    train_Loader = DataLoader(train_set)
    test_loader = DataLoader(test_set)
    extra_loader = DataLoader(extra_set)

    