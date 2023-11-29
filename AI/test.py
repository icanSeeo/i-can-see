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
from ResNet import ResNet, BasicBlock, Bottleneck, Identity

from ResNet import extract_feature_maps, config_model


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
    correct = 0
    feature_maps = None
    all_feature_maps = []

    all_feature_maps = [[] for _ in range(len(label_names))]  # [[], [], [], []
    combined_feature_maps = [[] for _ in range(len(label_names))]

    with torch.no_grad():
        for test_path in test_images_filepaths:
            img = Image.open(test_path).convert('RGB')
            _id = test_path.split('/')[-1].split('.')[0]
            label_idx = label_names.index(test_path.split('/')[-2])

            transform = ImageTransform(size, mean, std)
            img = transform(img, phase=phase)
            img = img.unsqueeze(0)
            img = img.to(device)

            model.eval()
            outputs = model(img)[0]

            preds_label_idx = torch.argmax(outputs, dim=1)
            # print('pred label', preds_label_idx)

            # 특징 맵 추출
            feature_maps = extract_feature_maps(model, img)
            all_feature_maps[preds_label_idx].append(feature_maps)

            if preds_label_idx == label_idx:
                correct += 1

        print(f"{phase} Test data ACC : {correct} / {total} = {correct / total}")

    return all_feature_maps


def reduce_dimensions(all_feature_maps, method='pca', n_components=3):
    for feature_maps in all_feature_maps:
        print(len(feature_maps), type(feature_maps))
    combined_all_feature_maps = [
        torch.cat(feature_maps, dim=0) for feature_maps in all_feature_maps]
    if method == 'pca':
        pca = PCA(n_components=n_components)
        reduced_all_features = [
            pca.fit_transform(combined_feature_maps.view(
                combined_feature_maps.size(0), -1).cpu().numpy())
            for combined_feature_maps in combined_all_feature_maps]
    elif method == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=10)
        reduced_all_features = [tsne.fit_transform(combined_feature_maps.view(combined_feature_maps.size(0), -1).cpu().numpy())
                                for combined_feature_maps in combined_all_feature_maps]
    else:
        raise ValueError('Wrong method name')

    return reduced_all_features


def plot_feature_maps(reduced_all_features, label_names, method='pca', n_components=3, dst_label_names=[]):

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')
        ax.set_title(f'{method.upper()} 3D Visualization')
        plt.legend(label_names)

        scatter_list = [
            ax.scatter(
                reduced_features[:, 0],
                reduced_features[:, 1],
                reduced_features[:, 2],
                label=label_name,
                alpha=0.5,
            )
            for label_name, reduced_features in zip(label_names, reduced_all_features) if label_name in dst_label_names]


        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return (*scatter_list, )

        ani = FuncAnimation(fig, update, frames=range(0, 360, 3), blit=True)
        ani.save('pca_3d_animation.gif', writer='pillow', fps=30)

    elif n_components == 2:
        plt.figure(figsize=(10, 10))
        for label_name, reduced_features in zip(label_names, reduced_all_features):
            if dst_label_names:
                if label_name in dst_label_names:
                    plt.scatter(reduced_features[:, 0],
                                reduced_features[:, 1], alpha=0.5, label=label_name)
            else:
                plt.scatter(reduced_features[:, 0],
                            reduced_features[:, 1], alpha=0.5)
        plt.title(f'{method.upper()} Visualization')
        plt.legend(label_names)

        plt.savefig(f'{method}_visualization.png')
    else:
        raise ValueError('Wrong n_components value')

# 차원축소 + 애니메이션 함수


def reduce_and_animate(feature_maps, method='pca', n_components=3):

    reduced_features = reduce_dimensions(feature_maps, method, n_components)

    # 3D 플롯
    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 산점도 표시
        scatter = ax.scatter(
            reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2])

        # 애니메이션 생성
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return scatter

        ani = FuncAnimation(fig, update, frames=range(0, 360, 3), blit=True)

    return reduced_features, ani


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    data_directory = './data/vc/train/'

    label_paths = sorted([os.path.join(data_directory, f)
                         for f in os.listdir(data_directory)])
    label_names = sorted([f for f in os.listdir(data_directory)])

    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    OUTPUT_DIM = len(label_names)
    model = ResNet(config_model(50), OUTPUT_DIM)
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    model_state_dict = torch.load("./resnet_train_v11.pt", map_location=device)
    model.load_state_dict(model_state_dict)

    _id = 0

    print(label_paths)
    print(label_names)
    # print('label : ', label_paths) # "./data/l4/bikes/", "./data/l4/cars/", "./data/l4/cats/", "./data/l4/dogs/"

    image_paths = [sorted([os.path.join(label_path, f) for f in os.listdir(
        label_path)]) for label_path in label_paths]
    # "./data/l4/bikes/0 ... 1000.jpg", "./data/l4/cars/0 ... 1000.jpg", "./data/l4/cats/0 ... 1000.jpg", "./data/l4/dogs/0 ... 1000.jpg"

    all_images_paths = [path for paths in image_paths for path in paths]
    print(len(all_images_paths))

    correct_images_filepaths = [
        i for i in all_images_paths if cv2.imread(i) is not None]

    # split train, val, test
    random.seed(42)
    random.shuffle(correct_images_filepaths)

    test_images_filepaths = correct_images_filepaths[:]
    print(len(test_images_filepaths))

    total = len(test_images_filepaths)

    phase1 = 'val'
    phase2 = 'random'
    phase3 = 'rotate'

    # print(model)

    num_ptrs = model.fc.in_features

    method = 'tsne'
    n_components = 3
    dst_label_names =  ['SUV', 'family sedan'] # ['SUV', 'racing car']

    all_feature_maps = inference_model(
        model, label_names, size, mean, std, total, phase1)
    reduced_all_features = reduce_dimensions(
        all_feature_maps, method=method, n_components=n_components)
    plot_feature_maps(reduced_all_features, label_names,
                      method=method, n_components=n_components, dst_label_names=dst_label_names)

    # inference_model(model, size, mean, std, total, phase2)
    # inference_model(model, size, mean, std, total, phase3)
