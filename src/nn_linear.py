import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                      download=True)
loader = DataLoader(datasets, batch_size=64, drop_last=True)

class MyLinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

linear = MyLinear()

for data in loader:
    img, target = data
    print(img.shape)
    output = torch.flatten(img)
    print(output.shape)
    output = linear(output)
    print(output.shape)