import torch.nn
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.Module import MyModule

# 准备数据集
train_datasets = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                              download=True)
test_datasets = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                             download=True)

train_datasets_size = len(train_datasets)
test_datasets_size = len(test_datasets)

# 加载数据集
batch_size = 64
train_dataloader = DataLoader(train_datasets, batch_size=batch_size, drop_last=True)
test_dataloader = DataLoader(test_datasets, batch_size=batch_size, drop_last=True)

# 创建网络模型
module = MyModule()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2 = 1 * (10)^-2 = 1 / 100 = 0.01
learning_rate = 1e-2
optim = torch.optim.SGD(module.parameters(), lr=learning_rate)

writer = SummaryWriter("../logs_train")

# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0

# 训练轮数
epoch = 10

for i in range(epoch):
    print(f"========第{i + 1}轮训练开始========")

    # 训练步骤
    for data in train_dataloader:
        imgs, target = data
        output = module(imgs)
        loss = loss_fn(output, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # 总训练次数
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"第{total_train_step}次训练，损失值：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    with torch.no_grad():
        # 总损失值
        total_test_loss = 0
        # 准确数
        total_accuracy = 0
        for data in test_dataloader:
            imgs, target = data
            output = module(imgs)
            loss = loss_fn(output, target)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy

    total_test_step += 1
    print(f"第{i + 1}次测试，损失值：{total_test_loss}")
    print(f"整体正确率：{total_accuracy / test_datasets_size}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_datasets_size, total_test_step)
    torch.save(module.state_dict(), f"train{i}.pth")
    print("模型已保存")

writer.close()
