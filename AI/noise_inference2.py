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

def reduce_dimensions(all_feature_maps, method='pca', compress=False):
    print("[*] 1. reduce dimensions")
    reduced_all_features = []
    
    for feature_maps in all_feature_maps:
        combined_feature_maps = torch.cat(tuple(feature_maps), dim=0)
        flattened_feature_maps = combined_feature_maps.view(
            combined_feature_maps.size(0), -1).cpu().numpy()
        
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=10)
        else:
            raise ValueError('Wrong method name')

        reduced_features = torch.from_numpy(reducer.fit_transform(flattened_feature_maps))
        mean_feature = torch.mean(reduced_features, dim=0)
        
        # print('flattened :', flattened_feature_maps.shape)
        # print('combined :', combined_feature_maps.shape)
        # print('flattened T:', flattened_feature_maps.T.shape)
        # print('reduced featuremap shape ', reduced_features.shape)
        # print('mean featuremap shape ', mean_feature.shape)
        # print()
        
        if compress:
            reduced_all_features.append(mean_feature.unsqueeze(0))
        else:
            reduced_all_features.append(reduced_features)
        
        
    return reduced_all_features


def plot_feature_maps(reduced_all_features, label_names, noise_weight, method='pca', model_name=''):
    print("[*] 2. ploting feature maps")
    result_fig = plt.figure(figsize=(10, 10))
    plt.xlim([-80, 80])      # X축의 범위: [xmin, xmax]
    plt.ylim([-80, 80])
    
    
    for (label_name, reduced_features) in zip(label_names, reduced_all_features):
        print('reduces feture shpoae : ', reduced_features.shape)
        plt.scatter(reduced_features[:, 0],
                reduced_features[:, 1], alpha=0.5, label=label_name)
    plt.title(f'{method.upper()} Visualization (noise : {noise_weight})')
    plt.legend()

    os.makedirs(f'./results/{model_name}/{method}', exist_ok=True)
    plt.savefig(f'./results/{model_name}/{method}/{noise_weight * 1000}.png')

def experience_image(image_path, model, noise_weight=0.1, compress=False, method='pca'):
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
    all_distances = [0]
    
    # noise 별로 featuremap 추출
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
        os.makedirs(f'./results/imgs/{noise_name}', exist_ok=True)
        plt.imsave(f'./results/imgs/{noise_name}/{noise_weight * 1000}.png', image)
        
        feature_maps = extract_feature_maps(model, noise)
        print("feature map shape :", feature_maps.shape)
        all_feature_maps.append(feature_maps)
        
        if noise_name != 'org':
            pass # all_distances.append(torch.dist(all_feature_maps[0], feature_maps))
        
    # 차원축소
    all_reduced_feature_maps = reduce_dimensions(all_feature_maps ,method, compress=compress)
    for reduced_feature_map in all_reduced_feature_maps[1:]:
        distance = torch.dist(all_reduced_feature_maps[0], reduced_feature_map)
        all_distances.append(distance)
        
    plot_feature_maps(all_reduced_feature_maps, noise_names, noise_weight=noise_weight, method=method, model_name='')
    
    return all_reduced_feature_maps, all_distances


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    base_model_name = "resnet18_baseline_AMF"  # PARAM1
    cbam_model_name = "resnet18_cbam03_AMF"  # PARAM1
    
    
    wandb.init(project='noise_exp', entity='joon0zo1022')
    wandb.run.name = 'final_f'


    data_directory = "./dataset/AMF/val"
    
    label_names = sorted([f for f in os.listdir(data_directory)])
    OUTPUT_DIM = len(label_names)
    
    # model file load
    
    base_model = ResNet(config_model(18), OUTPUT_DIM)
    cbam_model = ResNet(config_model(18), OUTPUT_DIM)
    
    cbam_model.layer3[-1].add_module('cbam', CBAM(1024))
    
    base_model_state_dict = torch.load(f"./pt_models/{base_model_name}.pt", map_location=device)
    cbam_model_state_dict = torch.load(f"./pt_models/{cbam_model_name}.pt", map_location=device)
    
    base_model.load_state_dict(base_model_state_dict)
    cbam_model.load_state_dict(cbam_model_state_dict)
    
    # reduce mothod
    method = 'tsne' # parma 1
    compress = False # parma 2
    
    # choose class
    choise_class = 26
    choise_label_name = label_names[choise_class]
    
    
    root, dirs, files = next(os.walk(f'{data_directory}/{choise_label_name}'))
    image_paths = sorted([os.path.join(root, f) for f in files])
    noise_names = ['org', 'light', 'dust', 'rotate', 'blur' ]
    
    # retreive
    for noise_weight in np.arange(0.0, 1.1, 0.01):
        
        noise_weight = round(noise_weight, 2)
        print("++++++++++++++++++++++++++++++++++++++")
        print(noise_weight)
        base_reduced_feature_maps, base_all_distance = experience_image(
            image_paths[0], base_model, noise_weight=noise_weight, compress=compress,
            method=method)
        plot_feature_maps(base_reduced_feature_maps, noise_names, noise_weight=noise_weight, method=method, model_name=base_model_name)
        
        cbam_reduced_feature_maps, cbam_all_distance = experience_image(
            image_paths[100], cbam_model, noise_weight=noise_weight, 
            compress=compress ,method=method)
        plot_feature_maps(cbam_reduced_feature_maps, noise_names, noise_weight=noise_weight, method=method, model_name=cbam_model_name)
        
        wandb.log({
            'base_light_distance': base_all_distance[1],
            'base_dust_distance': base_all_distance[2],
            'base_rotate_distance': base_all_distance[3],
            'base_blur_distance': base_all_distance[4],
            
            'cbam_light_distance': cbam_all_distance[1],
            'cbam_dust_distance': cbam_all_distance[2],
            'cbam_rotate_distance': cbam_all_distance[3],
            'cbam_blur_distance': cbam_all_distance[4],
        })
        
        base_reduced_feature_maps = [
            base_reduced_feature_maps[0], 
            torch.stack(
                (base_reduced_feature_maps[1], base_reduced_feature_maps[2], 
                 base_reduced_feature_maps[3], base_reduced_feature_maps[4])
            )
        ]
        base_reduced_feature_maps[1] = base_reduced_feature_maps[1].view(-1, 2)
        plot_feature_maps(base_reduced_feature_maps, ["org", "noise"], noise_weight=noise_weight, method=method, model_name=base_model_name + '-final')
        
        cbam_reduced_feature_maps = [
            cbam_reduced_feature_maps[0], 
            torch.stack(
                (cbam_reduced_feature_maps[1], cbam_reduced_feature_maps[2], 
                 cbam_reduced_feature_maps[3], cbam_reduced_feature_maps[4])
            )
        ]
        cbam_reduced_feature_maps[1] = cbam_reduced_feature_maps[1].view(-1, 2)
        plot_feature_maps(cbam_reduced_feature_maps, ["org", "noise"], noise_weight=noise_weight, method=method, model_name=cbam_model_name + '-final')
    
    