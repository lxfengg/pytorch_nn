import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(datasets, batch_size=64)

class My_Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(x)
        return x


relu = My_Relu()
writer = SummaryWriter("../logs_relu")

step = 0

for data in dataloader:
    img, target = data
    writer.add_images("input", img, step)

    output = relu(img)
    writer.add_images("output", output, step)
    step += 1

writer.close()