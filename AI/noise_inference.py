import os
import random
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import torch

from PIL import Image
from matplotlib.animation import FuncAnimation
from datetime import datetime
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from collections import namedtuple
from ImageTransform import ImageTransform
from ResNet import ResNet, BasicBlock, Bottleneck, Identity, CBAM
from tqdm import tqdm

from test import extract_feature_maps, config_model
from utils.noise_generators import *

# wandb
import wandb


device = "cuda"

def visualize_feature_maps(all_feature_maps):
    # 모든 이미지의 특징 맵을 하나의 배열로 결합
    combined_feature_maps = torch.cat(all_feature_maps, dim=0)

    # 특징 맵 시각화
    num_images = len(all_feature_maps)
    num_feature_maps = combined_feature_maps.size(1)

    fig, axes = plt.subplots(
        num_images, num_feature_maps, figsize=(15, 2*num_images))

    for i in range(num_images):
        for j in range(num_feature_maps):
            ax = axes[i, j]
            feature_map = combined_feature_maps[i, j].cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')  # 원하는 컬러맵 사용
            ax.axis('off')

    plt.show()


def inference_model(model, label_names, size, mean, std, total, phase):
    print("[*] 0. inference model")
    correct = 0
    feature_maps = None
    all_feature_maps = []

    all_feature_maps = [[] for _ in range(len(label_names))]  # [[], [], [], []
    combined_feature_maps = [[] for _ in range(len(label_names))]

    with torch.no_grad():
        for test_path in tqdm(test_images_filepaths):
            img = Image.open(test_path).convert('RGB')
            _id = test_path.split('\\')[-1].split('.')[0]
            label_idx = label_names.index(test_path.split('\\')[-2])

            transform = ImageTransform(size, mean, std)
            img = transform(img, phase=phase)
            img = img.unsqueeze(0)
            img = img.to(device)

            model.eval()
            outputs = model(img)[0]

            preds_label_idx = torch.argmax(outputs, dim=1)

            # 특징 맵 추출
            feature_maps = extract_feature_maps(model, img)
            all_feature_maps[preds_label_idx].append(feature_maps)


            if preds_label_idx == label_idx:
                correct += 1

        print(f"{phase} Test data ACC : {correct} / {total} = {correct / total}")

    return all_feature_maps


def reduce_dimensions(all_feature_maps, method='pca', n_components=3, compress=False):
    print("[*] 1. reduce dimensions")
    reduced_all_features = []
    
    for feature_maps in all_feature_maps:
        print('featuremap shape', feature_maps.shape)
        combined_feature_maps = torch.cat(tuple(feature_maps), dim=0)
        
        
        flattened_feature_maps = combined_feature_maps.view(
            combined_feature_maps.size(0), -1).cpu().numpy()
        
        
        if method == 'pca':
            pca = PCA(n_components=n_components)
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=10)
        else:
            raise ValueError('Wrong method name')

        reduced_features = torch.from_numpy(reducer.fit_transform(flattened_feature_maps))
        mean_feature = torch.mean(reduced_features, dim=0)
        
        print('flattened :', flattened_feature_maps.shape)
        print('combined :', combined_feature_maps.shape)
        print('flattened T:', flattened_feature_maps.T.shape)
        print('reduced featuremap shape ', reduced_features.shape)
        print('mean featuremap shape ', mean_feature.shape)
        print()
        
        if compress:
            reduced_all_features.append(mean_feature.unsqueeze(0))
        else:
            reduced_all_features.append(reduced_features)
        
        
    return reduced_all_features


def plot_feature_maps(reduced_all_features, noise_weight=0.0, method='pca', n_components=3, result_fig=None, dst_label_names=[]):
    print("[*] 2. ploting feature maps")
    # else:
    print(reduced_all_features)
    if n_components == 2:
        result_fig = plt.figure(figsize=(10, 10))
        for label_name, reduced_features in reduced_feature_map_dict.items():
            print(reduced_features[0].shape)
            plt.scatter(reduced_features[0][:, 0],
                        reduced_features[0][:, 1], alpha=0.5, label=label_name)
        plt.title(f'{method.upper()} Visualization')
        plt.legend()

        plt.savefig(f'./results/{method}_{noise_weight}_{model_name}_.png')
    else:
        raise ValueError('Wrong n_components value')


def experience_image(image_path, model, noise_weight=0.1, compress=False, method='pca', n_components=3, result_fig=None):
    transform = ImageTransform(noise_weight=noise_weight)  # PARAM2
    
    image = Image.open(image_path).convert('RGB')
    
    noise_names = ['org', 'light', 'dust', 'rotate', 'blur' ] # 'colorJit'
    noise_imgs = [
                    transform(image, phase='val'),
                    transform(image, phase='light'),
                    transform(image, phase='dust'),
                    transform(image, phase='rotate'),
                    transform(image, phase='blur'),
                    # transform(image, phase='colorJit')
                ]
    all_feature_maps = []
    
    model.eval()
    for noise, noise_name in zip(noise_imgs, noise_names):
        print('=====================================')
        noise = noise.unsqueeze(0)
        # noise = noise.to(device)
        outputs = model(noise)[0]
        preds_label_idx = torch.argmax(outputs, dim=1)
        print('noise name', noise_name)
        print('result (pred vs orgin)', label_names[preds_label_idx], 'vs', choise_label_name)
        
        
        
        image = transform.denormalize(noise.squeeze(0))
        plt.imsave(f'./results/{noise_name}_noise.png', image)
        
        feature_maps = extract_feature_maps(model, noise)
        print('feature map shape :' , feature_maps.shape)
        all_feature_maps.append(feature_maps)
        
        if noise_name != 'org':
            # distance 
            print('distance ver 1: ', torch.dist(all_feature_maps[0], feature_maps))
        
    all_reduced_feature_maps = reduce_dimensions(all_feature_maps ,method, n_components, compress=compress)
    print("all_reduced_feature_maps ======================")
    # print(all_reduced_feature_maps)
    
    print("distance ======================")
    all_distance = [0]
    for reduced_feature_map in all_reduced_feature_maps[1:]:
        # print(reduced_feature_map.shape)
        # print(all_reduced_feature_maps[0].shape)
        distance = torch.dist(all_reduced_feature_maps[0], reduced_feature_map)
        all_distance.append(distance)
        # similarity = nn.MSELoss()(all_reduced_feature_maps[0], reduced_feature_map)
        # print(similarity)
        
    
    # plot_feature_maps(all_reduced_feature_maps, noise_names, method, n_components, result_fig=result_fig)
    return all_reduced_feature_maps, all_distance


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    model_name = "resnet18_baseline_AMF"  # PARAM1
    
    # wandb.init(project='noise_exp', entity='joon0zo1022')
    # wandb.run.name = model_name


    data_directory = "./dataset/AMF/val"
    
    label_names = sorted([f for f in os.listdir(data_directory)])
    OUTPUT_DIM = len(label_names)
    model = ResNet(config_model(18), OUTPUT_DIM) 
    # model.layer3[-1].add_module('cbam', CBAM(1024)) # PARAM1.5
    model_state_dict = torch.load(f"./pt_models/{model_name}.pt", map_location=device)
    model.load_state_dict(model_state_dict)
    
    choise_class = 20
    method = 'tsne'
    
    choise_label_name = label_names[choise_class]
    root, dirs, files = next(os.walk(f'{data_directory}/{choise_label_name}'))
    image_paths = sorted([os.path.join(root, f) for f in files])
    noise_names = ['org', 'light', 'dust', 'rotate', 'blur' ]
    
    print
    for noise_weight in np.arange(0.0, 1.0, 0.1):
        reduced_feature_map_dict = {noise_name : []  for noise_name in noise_names}
        distance_dict = {noise_name : []  for noise_name in noise_names}
        
        for image_path in image_paths[:1]: # 이미지 하나에 여러 노이즈 적용 후 featuremap 추출
            all_reduced_feature_maps, all_distance = experience_image(image_path, model, noise_weight=noise_weight, compress=False ,method=method, n_components=2)
            
            # 각각의 노이즈에 대한 featuremap과 distance를 저장
            for noise_name, reduced_feature_map in zip(noise_names, all_reduced_feature_maps):
                reduced_feature_map_dict[noise_name].append(reduced_feature_map)
                
            for noise_name, distance in zip(noise_names, all_distance):
                distance_dict[noise_name].append(distance)
        
        print('first distance : ', distance_dict)
        # 각 노이즈에 대한 featuremap과 distance를 평균
        for noise_name in noise_names:
            print(f'noise {noise_name}: ', reduced_feature_map_dict[noise_name])
            print(type(reduced_feature_map_dict[noise_name]))
            mean_reduced_feature_map = reduced_feature_map_dict[noise_name][0] # torch.mean(reduced_feature_map_dict[noise_name], dim=0)
            mean_distance = distance_dict[noise_name][0] # torch.mean(distance_dict[noise_name], dim=0)
            
        # wandb.log({
        #     'light_distance': distance_dict['light'][0],
        #     'dust_distance': distance_dict['dust'][0],
        #     'rotate_distance': distance_dict['rotate'][0],
        #     'blur_distance': distance_dict['blur'][0],
        # })
        
        
        plot_feature_maps(reduced_feature_map_dict, noise_weight=noise_weight, method=method, n_components=2, result_fig=None)
            
    
    # print(reduced_feature_map_dict)
    # print(distance_dict)
    
    
    
    