# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import namedtuple
from AI.classification.model.ResNet import BasicBlock, Bottleneck

def config_model(n):
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    if n == 18:
        resnet18_config = ResNetConfig(block = BasicBlock,
                                    n_blocks = [2,2,2,2],
                                    channels = [64, 128, 256, 512])
        return resnet18_config
    elif n == 34:
        resnet34_config = ResNetConfig(block = BasicBlock,
                                    n_blocks = [3,4,6,3],
                                    channels = [64, 128, 256, 512])
        return resnet34_config
    elif n == 50:
        resnet50_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 6, 3],
                                    channels = [64, 128, 256, 512])
        return resnet50_config
    elif n == 101:
        resnet101_config = ResNetConfig(block = Bottleneck,
                                        n_blocks = [3, 4, 23, 3],
                                        channels = [64, 128, 256, 512])
        return resnet101_config
    elif n == 152:
        resnet152_config = ResNetConfig(block = Bottleneck,
                                        n_blocks = [3, 8, 36, 3],
                                        channels = [64, 128, 256, 512])
        return resnet152_config