import torch
import torch.nn.functional as funcs


class ToyNet1(torch.nn.Module):
    def __init__(self):
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
        torch.conv5 = torch.nn.Conv2d(
            128 * 3, 128 * 6, kernel_size=3, padding=1,
            stride=2, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(128 * 6)
        torch.conv6 = torch.nn.Conv2d(
            128 * 6, 128 * 6, kernel_size=3,
            padding=1, bias=False, stride=2
        )
        torch.conv7 = torch.nn.Conv2d(
            128 * 6, 128 * 6, kernel_size=3,
            padding=1, bias=False, stride=2
        )
        self.bn4 = torch.BatchNorm2d(128 * 6)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        out = funcs.relu(self.bn1(self.conv1(x)))
        out = self.conv3(self.conv2(out))
        out = funcs.relu(self.bn2(out))
        