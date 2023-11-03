import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


class MyModule(nn.Module):

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
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.seq(x)
        return output

if __name__ == '__main__':

    # 简单验证模型输出是否正确
    module = MyModule()
    input = torch.ones((64, 3, 32, 32))
    output = module(input)
    print(input.shape)
    print(output.shape)