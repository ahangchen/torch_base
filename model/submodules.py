import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_bn_relu(in_dim, out_dim, kernel, stride=1, pad=0, dilate=1, group=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad, dilate, group),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )


def up_conv2d(in_dim, out_dim, kernel=3, pad=1, up_scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=up_scale, mode='nearest'),
        nn.Conv2d(in_dim, out_dim, kernel, padding=pad)
    )

