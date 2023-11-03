import torch
import torchvision
from nn_sequential import *
# 加载1
vgg16_1 = torch.load("vgg16_method1.pth")
print(vgg16_1)

# 加载3
# vgg16_2 = torch.load("vgg16_method2.pth")
# print(vgg16_2)
vgg_2 = torchvision.models.vgg16()
vgg_2.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg_2)

# 自己模型
myseq = torch.load("myseq.pth")
print(myseq)

seq = MySeq()
seq.load_state_dict(torch.load("myseq1.pth"))
print(myseq)