import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import json
import cv2
import numpy as np

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
