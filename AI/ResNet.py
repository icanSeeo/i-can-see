import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BasicBlock(nn.Module):    
        expansion = 1
        
        def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
            super().__init__()                
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                                stride = stride, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)        
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                                stride = 1, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(out_channels)        
            self.relu = nn.ReLU(inplace = True)
            
            if downsample:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                                stride = stride, bias = False)
                bn = nn.BatchNorm2d(out_channels)
                downsample = nn.Sequential(conv, bn)
            else:
                downsample = None        
            self.downsample = downsample
            
        def forward(self, x):       
            i = x       
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)        
            x = self.conv2(x)
            x = self.bn2(x)
            
            if self.downsample is not None:
                i = self.downsample(i)
                            
            x += i
            x = self.relu(x)
            
            return x
        
class Bottleneck(nn.Module):    
        expansion = 4
        
        def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
            super().__init__()    
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)        
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(out_channels)        
            self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                                stride = 1, bias = False)
            self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)        
            self.relu = nn.ReLU(inplace = True)
            
            if downsample:
                conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                                stride = stride, bias = False)
                bn = nn.BatchNorm2d(self.expansion * out_channels)
                downsample = nn.Sequential(conv, bn)
            else:
                downsample = None            
            self.downsample = downsample
            
        def forward(self, x):        
            i = x        
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)        
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)        
            x = self.conv3(x)
            x = self.bn3(x)
                    
            if self.downsample is not None:
                i = self.downsample(i)
                
            x += i
            x = self.relu(x)
        
            return x
        
class ResNet(nn.Module):
        def __init__(self, config, output_dim, zero_init_residual=False):
            super().__init__()
                    
            block, n_blocks, channels = config
            self.in_channels = channels[0]            
            assert len(n_blocks) == len(channels) == 4
            
            self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
            self.bn1 = nn.BatchNorm2d(self.in_channels)
            self.relu = nn.ReLU(inplace = True)
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            
            self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
            self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
            self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
            self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(self.in_channels, output_dim)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
            
        def get_resnet_layer(self, block, n_blocks, channels, stride = 1):   
            layers = []        
            if self.in_channels != block.expansion * channels:
                downsample = True
            else:
                downsample = False
            
            layers.append(block(self.in_channels, channels, stride, downsample))
            
            for i in range(1, n_blocks):
                layers.append(block(block.expansion * channels, channels))

            self.in_channels = block.expansion * channels            
            return nn.Sequential(*layers)
            
        def forward(self, x):        
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)        
            x = self.avgpool(x)
            h = x.view(x.shape[0], -1)
            x = self.fc(h)        
            return x, h

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_gate = ChannelAttention(in_planes, ratio)
        self.spatial_gate = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_gate(x) * x
        x = self.spatial_gate(x) * x
        return x