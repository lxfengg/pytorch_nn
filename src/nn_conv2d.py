import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(datasets, batch_size=64)

class MyConv2(nn.Module):
    def __init__(self):
        super(MyConv2, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

conv = MyConv2()
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = conv(imgs)

    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images("inputs", imgs, step)
    writer.add_images("outputs", outputs, step)
    step += 1

    print(imgs.shape)
    print(outputs.shape)

writer.close()