import torch
import torch.nn.functional as funcs


class ToyNet1(torch.nn.Module):
    def __init__(self, num_class=10):
        super(ToyNet1, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=5, stride=1,
            padding=2, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(
            64, 64, kernel_size=(3, 3),
            padding=1, stride=2, bias=False
        )
        self.conv3 = torch.nn.Conv2d(
            64, 128, kernel_size=3,
            padding=1, stride=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv4_1 = torch.nn.Conv2d(
            128, 128, kernel_size=1, padding=0,
            stride=1, bias=False
        )
        self.conv4_2 = torch.nn.Conv2d(
            128, 128, kernel_size=3, padding=1,
            stride=1, bias=False
        )
        self.conv4_3 = torch.nn.Conv2d(
            128, 128, kernel_size=5, padding=2,
            stride=1, bias=False
        )
        self.conv5 = torch.nn.Conv2d(
            128 * 3, 128 * 6, kernel_size=3, padding=1,
            stride=1, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(128 * 6)
        self.conv6 = torch.nn.Conv2d(
            128 * 6, 128 * 6, kernel_size=3,
            padding=1, bias=False, stride=2
        )
        self.bn4 = torch.nn.BatchNorm2d(128 * 6)
        self.conv7 = torch.nn.Conv2d(
            128 * 6, 128 * 6, kernel_size=3,
            padding=1, bias=False, stride=2
        )
        self.bn5 = torch.nn.BatchNorm2d(128 * 6)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=4)
        self.fc = torch.nn.Linear(128 * 6, num_class)

    def forward(self, x):
        out = funcs.relu(self.bn1(self.conv1(x)))
        out = self.conv3(self.conv2(out))
        out = funcs.relu(self.bn2(out))
        out1 = self.conv4_1(out)
        out2 = self.conv4_2(out)
        out3 = self.conv4_3(out)
        out = torch.cat([out1, out2, out3], 1)
        out = self.conv5(out)
        out = funcs.relu(self.bn3(out))
        out = self.conv6(out)
        out = funcs.relu(self.bn4(out))
        out = funcs.relu(self.bn5(self.conv7(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    x = torch.ones([2, 3, 32, 32])
    model = ToyNet1()
    out = model(x)
    print(out.shape)
