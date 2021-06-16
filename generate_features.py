from random import shuffle
from torchvision.datasets import SVHN
from skimage import feature as feat
from tqdm import tqdm
import numpy as np
import pickle


def Generate_Hog(img):
    return feat.hog(
        img, pixels_per_cell=[4, 4],
        cells_per_block=[2, 2], orientations=8
    )


def Get_Dataset():
    train_set = SVHN('../data', split='train', download=False)
    test_set = SVHN('../data', split='test', download=False)
    extra_set = SVHN('../data', split='extra', download=False)
    return train_set, test_set, extra_set


if __name__ == '__main__':
    train_set, test_set, extra_set = Get_Dataset()
    train_len = len(train_set)

    train_features, train_labels = [], []
    train_figs = []
    for i in tqdm(range(train_len)):
        img, label = train_set[i]
        fea = Generate_Hog(img)
        train_features.append(fea)
        train_labels.append(label)
        train_figs.append(img)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    with open('train_figs.pkl', 'wb') as Fout:
        pickle.dump(train_figs, Fout)

    test_features, test_labels = [], []
    test_figs = []

    test_len = len(test_set)
    for i in tqdm(range(test_len)):
        img, label = test_set[i]
        fea = Generate_Hog(img)
        test_features.append(fea)
        test_labels.append(label)
        test_figs.append(img)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    np.save('test_labels.npy', test_labels)
    np.save('test_features.npy', test_features)
    with open('test_figs.pkl', 'wb') as Fout:
        pickle.dump(test_figs, Fout)

    Lab2Col = {}
    extra_len = len(extra_set)
    for i in tqdm(range(extra_len)):
        img, label = extra_set[i]
        if label not in Lab2Col:
            Lab2Col[label] = []
        Lab2Col[label].append(i)

    for k in Lab2Col.keys():
        shuffle(Lab2Col[k])

    extra_features, extra_labels = [], []
    extra_figs = []
    for label, v in Lab2Col.items():
        for idx in tqdm(v[:5000]):
            img, lab = extra_set[idx]
            fea = Generate_Hog(img)
            extra_features.append(fea)
            extra_labels.append(lab)

    Idxlist = list(range(len(extra_features)))
    shuffle(Idxlist)
    ep, fp = [], []
    for i in Idxlist:
        ep.append(extra_features[i])
        fp.append(extra_labels[i])
        extra_figs.append(extra_set[i][0])

    ep = np.array(ep)
    fp = np.array(fp)
    np.save('extra_features.npy', ep)
    np.save('extra_labels.npy', fp)
    with open("extra_figs.pkl", 'wb') as Fout:
        pickle.dump(extra_figs, Fout)
