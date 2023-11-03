import torchvision.models

vgg_16_false = torchvision.models.vgg16()
vgg_16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

print(vgg_16_false)
print(vgg_16_true)