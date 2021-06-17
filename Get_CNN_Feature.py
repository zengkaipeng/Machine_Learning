import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model

        def save_output(module, Input, output):
            self.buffer = output
        self.model.avgpool.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('result/CNN1/epoch_20.model')
    test_feautures = torch.load("CNN_Data/test_features.tov")
    test_labels = torch.load('CNN_Data/test_labels.tov')
    test_set = TensorDataset(test_feautures, test_labels)
    bs = 64
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    net = Net(model)
    net.eval()
    net.cuda()
    Tot_Features = []
    tot_labels = []
    for Idx, (data, labels) in enumerate(tqdm(test_loader)):
        batchlen = len(data)
        data = data.to(device)
        result = net(data)
        result = result.detach().cpu()
        result = result.reshape((batchlen, -1))
        Tot_Features.append(result)
        tot_labels.append(labels)

    Tot_Features = torch.cat(Tot_Features, 0)
    tot_labels = torch.cat(tot_labels, 0)
    print(Tot_Features.shape)
    torch.save(Tot_Features, 'tot_test_features.tov')
    torch.save(tot_labels, 'tot_test_labels.tov')
