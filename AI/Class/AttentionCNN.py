import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import CBAM

class AttentionCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(AttentionCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cbam = CBAM(in_planes=...)  # Specify the number of input channels to CBAM
        self.fc = nn.Linear(..., num_classes)  # Specify the appropriate input size for the fully connected layer

    def forward(self, x):
        out = self.conv_block(x)
        out = self.cbam(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out