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
    # 주어진 범위 내에서 밝기 조정
    scale_factor = np.random.uniform(*scale_range) 
    noisy = image * scale_factor
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_dust_and_particles(image, dust_prob=0.1, particle_prob=0.05):
   # 먼지와 파티클 추가
   noisy = np.copy(image)
   dust_mask = np.random.rand(*image.shape[:2]) < dust_prob
   dusty_pixels = np.random.randint(150, 200, image.shape[2])
   noisy[dust_mask] = dusty_pixels

   particle_mask = np.random.rand(*image.shape[:2]) < particle_prob
   particle_pixels = np.random.randint(50, 100, image.shape[2])
   noisy[particle_mask] = particle_pixels

   return noisy.astype(np.uint8)

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

def visualize_features(features, images, method='tsne'):
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        projected = tsne.fit_transform(features) 
   
    plt.figure(figsize=(10,10))
    plt.scatter(projected[:,0], projected[:,1], c='blue', alpha=0.5)

    for i, txt in enumerate(images):
        plt.annotate(txt, (projected[i,0], projected[i,1])) 

    plt.gca().set_aspect('equal', 'datalim')
    plt.title('t-SNE visualization of image features', fontsize=20)

# 이미지 한장 로드
img_path = 'image.jpg' 
orig_img = cv2.imread(img_path)

# 노이즈 종류와 강도 정의
noise_types = ['lighting', 'dust', 'blur']
noise_levels = [0.2, 0.5, 0.8] 

# 노이즈 이미지 생성
noisy_images = []
for noise_type in noise_types:
    for level in noise_levels:
        if noise_type == 'lighting':
            noisy_img = add_lighting_variation(orig_img, level) 
        elif noise_type == 'dust':
            noisy_img = add_dust_and_particles(orig_img, dust_prob=level)
        # ...
        noisy_images.append(noisy_img)

# 모델 정의            
model = models.resnet18(pretrained=True) 
model.eval()

# 노이즈 이미지로부터 피쳐 맵 추출
image_features = extract_features(model, noisy_images)

# t-SNE 시각화
image_labels = [f'{t}_{l}' for t in noise_types for l in noise_levels] 
visualize_features(image_features, image_labels)