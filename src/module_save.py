import torch
import torchvision
from nn_sequential import MySeq


vgg_false = torchvision.models.vgg16()
vgg_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

# 完整模型
torch.save(vgg_false, "vgg16_method1.pth")
# 模型参数
torch.save(vgg_true.state_dict(), "vgg16_method2.pth")
# 自己模型
seq = MySeq()
torch.save(seq, "myseq.pth")
torch.save(seq.state_dict(), "myseq1.pth")