import torchvision
import torch.nn as nn

from model.submodules import conv2d_bn_relu


class LightFcn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.custom_module = conv2d_bn_relu(3, 3, 3)
        self.fcn = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()

    def forward(self, img):
        x = self.custom_module(img)
        y = self.fcn(x)
        return y