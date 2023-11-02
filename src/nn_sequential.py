import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class MySeq(nn.Module):

    def __init__(self):
        super().__init__()
        # 奇数大小卷积核，可以通过 (kernel_size - 1) / 2计算padding
        # 偶数大小卷积核，可以通过 kernel_size / 2向上取整，输出size至少与输入相等
        # 也可以通过(kernel_size - 1) / 2后向下取整，减少计算量
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d有完整计算公式
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # # 计算线性层输入，可以先运行，查看flatten后输出size，即为线性层输入
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.seq(x)
        return x

writer = SummaryWriter("../logs_seq")

seq = MySeq()
print(seq)
input = torch.ones((64, 3, 32, 32))
print(input.shape)
output = seq(input)
print(output.shape)
writer.add_graph(seq, input)

writer.close()