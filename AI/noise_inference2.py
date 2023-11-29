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
import matplotlib.animation as animation

# 노이즈 추가 함수들
def add_lighting_variation(image, scale_range=(0.5, 1.5)):
    scale_factor = np.random.uniform(*scale_range)
    noisy = image * scale_factor
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_dust_and_particles(image, dust_prob=0.1, particle_prob=0.05):
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
    with torch.no_grad():
        for img in images:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)
            output = model(img_tensor)
            features.append(output.detach().cpu().numpy().squeeze())
    return np.array(features)

def visualize_features(orig_projected, noisy_projected):
   
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    orig_scatter = ax.scatter(orig_projected[:,0], orig_projected[:,1], c='red')
    noisy_scatter = ax.scatter(noisy_projected[:,0], noisy_projected[:,1], c='blue', alpha=0.5)

    def anim(i):
        ax.view_init(elev=10., azim=i)
        return orig_scatter, noisy_scatter

    anim = animation.FuncAnimation(fig, anim, frames=360, interval=100)
    return anim

# 주어진 이미지 로드
img_path ='./data/vc/train/taxi/07791caad841c29378938e7ab8a7b2ad.jpg'
orig_img = cv2.imread(img_path)

# 노이즈 추가 
noise_types = ['lighting', 'dust', 'blur']
noise_levels = [0.2, 0.5, 0.8]

noisy_images = []
for t in noise_types:
    for l in noise_levels:
        if t == 'lighting': 
            noisy_img = add_lighting_variation(orig_img, l)
        elif t == 'dust':
            noisy_img = add_dust_and_particles(orig_img, dust_prob=l)
            
        noisy_images.append(noisy_img)
        
# 모델 정의
model = models.resnet18(pretrained=True)
model.eval() 

# 피처 추출
orig_features = 

# 애니메이션
anim = visualize_features(orig_projected, noisy_projected)
anim.save('feature_animation.gif')extract_features(model, [orig_img])
noisy_features = extract_features(model, noisy_images)

# 차원 축소
tsne = TSNE(n_components=2, random_state=42)
orig_projected = tsne.fit_transform(orig_features)
noisy_projected = tsne.fit_transform(noisy_features)