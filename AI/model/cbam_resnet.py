import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import json
import cv2
import numpy as np

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAMModule(nn.Module):
    def __init__(self, in_channels):
        super(CBAMModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out

class ResNetWithCBAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetWithCBAM, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cbam1 = CBAMModule(256)
        self.cbam2 = CBAMModule(512)
        self.cbam3 = CBAMModule(1024)
        self.cbam4 = CBAMModule(2048)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        print('x shape', x.shape)
        
        print('x layer1 shape', self.layer1(x).shape)
        x = self.cbam1(self.layer1(x))
        print('x cbam1 shape', x.shape)
        
        print('x layer2 shape', self.layer2(x).shape)
        x = self.cbam2(self.layer2(x))
        print('x cbam2 shape', x.shape)
        
        print('x layer3 shape', self.layer3(x).shape)
        x = self.cbam3(self.layer3(x))
        print('x cbam3 shape', x.shape)
        
        print('x layer4 shape', self.layer4(x).shape)
        x = self.cbam4(self.layer4(x))
        print('x cbam4 shape', x.shape)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        
        # Calculate attention map (channel-wise and spatial-wise)
        channel_attention = x.mean(dim=(2, 3), keepdim=True)  # Calculate channel-wise attention
        spatial_attention = x.sum(dim=1, keepdim=True)  # Calculate spatial-wise attention

        # Combine channel-wise and spatial-wise attention
        attention_map = channel_attention * spatial_attention

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, attention_map

def resnet50_cbam(num_classes=1000):
    return ResNetWithCBAM(Bottleneck, [3, 4, 6, 3], num_classes)

# 이미지를 전처리하는 함수 정의
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch

# 클래스 인덱스를 클래스명으로 변환하는 함수 정의
def load_class_names():
    with open('imagenet_classes.json') as f:
        class_names = json.load(f)
    return class_names

# GPU를 사용할 수 있는 경우 GPU로 모델 및 입력 데이터를 이동하는 함수 정의
def prepare_model_and_input(model, input_batch):
    if torch.cuda.is_available():
        model.to('cuda')
        input_batch = input_batch.to('cuda')
    return model, input_batch

# 추론을 수행하는 함수 정의
def perform_inference(model, input_batch):
    with torch.no_grad():
        output, attention_map = model(input_batch)
    return output, attention_map

# 결과를 해석하는 함수 정의
def interpret_output(output, class_names):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(class_names[top5_catid[i]], top5_prob[i].item())

def overlay_attention_map(original_image, attention_map, output_path, alpha=0.5):

    # Normalize attention map values to range [0, 255]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) * 255
    # attention_map = attention_map.astype(np.uint8)

    # Resize attention map to match the original image size
    attention_map = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))

    # Apply colormap to attention map for better visualization (optional)
    attention_colormap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

    # Blend attention map with the original image
    overlaid_image = cv2.addWeighted(original_image, 1-alpha, attention_colormap, alpha, 0)

    # Save the overlaid image
    cv2.imwrite(output_path, overlaid_image)

# 이미지 경로 설정
image_path = 'piano.jpg'

# 이미지 전처리 및 모델 준비
input_batch = preprocess_image(image_path)
model = ResNetWithCBAM(Bottleneck, [3, 4, 6, 3], num_classes=1000)

pretrained_resnet = resnet50(pretrained=True)
model.load_state_dict(pretrained_resnet.state_dict(), strict=False)
model.eval()
model, input_batch = prepare_model_and_input(model, input_batch)

# 추론 수행
output, attention_map = perform_inference(model, input_batch)

# 클래스 인덱스를 클래스명으로 변환
class_names = load_class_names()

# 결과 해석 및 출력
interpret_output(output, class_names)

original_image = cv2.imread('piano.jpg')
print(original_image.shape)
overlay_attention_map(original_image, attention_map.numpy(), 'output_image.jpg')