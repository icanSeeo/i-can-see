import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from sklearn.manifold import TSNE

# 노이즈 추가 함수들

def add_lighting_variation(image, scale_range=(0.5, 1.5)):
  # 생략

def add_dust_and_particles(image, dust_prob=0.1, particle_prob=0.05):
  # 생략
  
# 기타 노이즈 함수들   

def extract_features(model, images):
    features = []
    model.eval()
    with torch.no_grad():
        for img in images:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) 
            output = model(img_tensor)
            features.append(output.detach().cpu().numpy().squeeze())
    return np.array(features)

def visualize_features(orig_features, noisy_features):
    # 차원 축소
    tsne = TSNE(n_components=2, random_state=42)
    orig_projected = tsne.fit_transform(orig_features)
    noisy_projected = tsne.fit_transform(noisy_features)

    # 차원축소된 피처들 시각화 
    plt.figure(figsize=(10,10))
    plt.scatter(orig_projected[:,0], orig_projected[:,1], c='red', label='original')
    plt.scatter(noisy_projected[:,0], noisy_projected[:,1], c='blue', alpha=0.5, label='noisy')
    plt.legend()
    plt.title('t-SNE plot', fontsize=20)

    # 시각화 결과 저장
    plt.savefig('tsne_plot.png')

# 이미지 한장 로드
img_path = 'data/vc/train/taxi/07791caad841c29378938e7ab8a7b2ad.jpg'
orig_img = cv2.imread(img_path)  

# 노이즈 파라미터 정의
noise_types = ['lighting', 'dust', 'blur'] 
noise_levels = [0.2, 0.5, 0.8]

# 노이즈 이미지 생성
noisy_images = []
for noise_type in noise_types:
    for level in noise_levels:
       # 노이즈 함수 호출
        
# 모델 정의
model = models.resnet18(pretrained=True)
model.eval()

# 피처 맵 추출
orig_features = extract_features(model, [orig_img])
noisy_features = extract_features(model, noisy_images)

# 시각화
visualize_features(orig_features, noisy_features)