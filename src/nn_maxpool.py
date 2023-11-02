import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype = torch.float)
#
# input = torch.reshape(input, (-1, 1, 5, 5))

datasets = torchvision.datasets.CIFAR10("../data", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(datasets, batch_size=64)

class MyPool(nn.Module):
    def __init__(self):
        super(MyPool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x

mp = MyPool()
# output = mp(input)
# print(output)
writer = SummaryWriter("../logs_maxpool")

step = 0
for data in dataloader:
    img, target = data
    writer.add_images("input", img, step)
    output = mp(img)
    writer.add_images("output", output, step)
    step += 1
writer.close()