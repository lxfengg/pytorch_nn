import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


class MySeq(nn.Module):

    def __init__(self):
        super().__init__()

        self.seq = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        return x

datasets = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(datasets, batch_size=1)
loss = nn.CrossEntropyLoss()
seq = MySeq()
for data in dataloader:
    imgs, targets = data
    output = seq(imgs)
    result_loss = loss(output, targets)
    # result_loss.backward()
    print(result_loss)