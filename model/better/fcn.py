import torchvision
import torch.nn as nn

from model.submodules import conv2d_bn_relu


class Resnet101Fcn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.custom_module = conv2d_bn_relu(3, 3, 3)
        self.fcn = torchvision.models.segmentation.fcn_resnet101()

    def forward(self, img):
        x = self.custom_module(img)
        y = self.fcn(x)
        return y