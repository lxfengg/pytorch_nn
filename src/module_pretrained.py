import torchvision.models
from torch import nn

vgg_16_false = torchvision.models.vgg16()
vgg_16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

print(vgg_16_false)

# vgg_16_true.add_module("add", nn.Linear(1000, 10))
vgg_16_true.classifier.add_module("add", nn.Linear(1000, 10))
print(vgg_16_true)

vgg_16_false.classifier[6] = nn.Linear(1000, 10)
print(vgg_16_false)