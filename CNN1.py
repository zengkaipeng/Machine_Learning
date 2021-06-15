from CNN_Module import ToyNet1
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
from tqdm import tqdm
import numpy as np


def Test(dataloader, model, device, totlen):
    Lossfun = torch.nn.CrossEntropyLoss()
    Loss = torch.tensor(0.0)
    Corr = torch.tensor(0)
    for Idx, (data, label) in enumerate(tqdm(dataloader)):
        data = data.to(device)
        label = label.to(device)
        result = model(data)
        Loss += Lossfun(result, label).item()
        Predict = torch.softmax(result, dim=1).argmax(axis=1)
        Corr += torch.sum(Predict == label)

    return Loss.item(), Corr.item() / totlen


if __name__ == '__main__':
    train_features = torch.load('CNN_Data/train_features.tov')
    train_labels = torch.load('CNN_Data/train_labels.tov')
    test_feautures = torch.load("CNN_Data/test_features.tov")
    test_labels = torch.load('CNN_Data/test_labels.tov')

    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_feautures, test_labels)
    trainlen = len(train_features)
    testlen = len(test_feautures)

    model = ToyNet1(10)
    Lossfun = torch.nn.CrossEntropyLoss()
    bs = 256
    init_lr = 1e-3
    train_loader = DataLoader(train_set, batch_size=bs)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr,
        betas=(0.9, 0.999), eps=1e-8
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ep in range(50):
        torch.cuda.empty_cache()
        Loss = torch.tensor(0.0)
        Corr = torch.tensor(0)
        for Idx, (data, label) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            label = label.to(device)
            result = model(data)
            Losv = Lossfun(result, label)
            Predict = torch.softmax(result, dim=1).argmax(axis=1)
            Corr += torch.sum(Predict == label)
            Losv.backward()
            Loss += Losv.item()
            optimizer.step()
            lr_scheduler.step()

        print("Epoch = {}, lr = {}\nTrainAcc = {} TrainLoss={}".format(
            epoch, lr_scheduler.get_last_lr(),
            Corr.item() / trainlen, Loss.item()
        ))
