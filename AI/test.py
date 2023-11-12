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

def visualize_feature_maps(all_feature_maps):
    # 모든 이미지의 특징 맵을 하나의 배열로 결합
    combined_feature_maps = torch.cat(all_feature_maps, dim=0)

    # 특징 맵 시각화
    num_images = len(all_feature_maps)
    num_feature_maps = combined_feature_maps.size(1)

    fig, axes = plt.subplots(num_images, num_feature_maps, figsize=(15, 2*num_images))

    for i in range(num_images):
        for j in range(num_feature_maps):
            ax = axes[i, j]
            feature_map = combined_feature_maps[i, j].cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')  # 원하는 컬러맵 사용
            ax.axis('off')

    plt.show()

def extract_feature_maps(model, test_image):
    desired_layer = 'layer4' 
    
    # 특징 맵 추출
    feature_maps = None
    hooks = []
    
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output
    
    for name, layer in model.named_modules():
        if name == desired_layer:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    with torch.no_grad():
        model(test_image)
    
    for hook in hooks:
        hook.remove()
    
    return feature_maps

def config_model(n):
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    if n == 18:
        resnet18_config = ResNetConfig(block = BasicBlock,
                                    n_blocks = [2,2,2,2],
                                    channels = [64, 128, 256, 512])
        return resnet18_config
    elif n == 34:
        resnet34_config = ResNetConfig(block = BasicBlock,
                                    n_blocks = [3,4,6,3],
                                    channels = [64, 128, 256, 512])
        return resnet34_config
    elif n == 50:
        resnet50_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 6, 3],
                                    channels = [64, 128, 256, 512])
        return resnet50_config
    elif n == 101:
        resnet101_config = ResNetConfig(block = Bottleneck,
                                        n_blocks = [3, 4, 23, 3],
                                        channels = [64, 128, 256, 512])
        return resnet101_config
    elif n == 152:
        resnet152_config = ResNetConfig(block = Bottleneck,
                                        n_blocks = [3, 8, 36, 3],
                                        channels = [64, 128, 256, 512])
        return resnet152_config
def inference_model(model, size, mean, std, total, phase):
    correct = 0
    id_list.clear()
    pred_list.clear()
    feature_maps = None
    all_feature_maps = []
    dog_feature_maps = []
    cat_feature_maps = []

    with torch.no_grad():
        for test_path in test_images_filepaths:
            img = Image.open(test_path).convert('RGB')
            _id =test_path.split('\\')[-1].split('.')[0]
            label = label_dic[test_path.split('\\')[-2]]
            
            transform = ImageTransform(size, mean, std)
            img = transform(img, phase=phase)
            img = img.unsqueeze(0)
            img = img.to(device)

            model.eval()
            outputs = model(img)
            preds = F.softmax(outputs[0], dim=1)[:, 1].tolist()[0]

            if preds > 0.5:
                preds = 1
            else:
                preds = 0
            
            # 특징 맵 추출
            feature_maps = extract_feature_maps(model, img)
            if label == 1:
                dog_feature_maps.append(feature_maps)
            else:
                cat_feature_maps.append(feature_maps)

            if preds == label:
                correct += 1
            id_list.append(_id)
            pred_list.append(preds)

    print(f"{phase} Test data ACC : {correct} / {total} = {correct / total}")

    # 강아지와 고양이 피쳐 맵을 하나의 배열로 결합
    combined_dog_feature_maps = torch.cat(dog_feature_maps, dim=0)
    combined_cat_feature_maps = torch.cat(cat_feature_maps, dim=0)

    # 각 클래스에 대해 t-SNE를 사용하여 3차원으로 축소
    # tsne = TSNE(n_components=2, random_state=42)
    # reduced_dog_features = tsne.fit_transform(combined_dog_feature_maps.view(combined_dog_feature_maps.size(0), -1).cpu().numpy())
    # reduced_cat_features = tsne.fit_transform(combined_cat_feature_maps.view(combined_cat_feature_maps.size(0), -1).cpu().numpy())

    pca = PCA(n_components=3)
    reduced_dog_features = pca.fit_transform(combined_dog_feature_maps.view(combined_dog_feature_maps.size(0), -1).cpu().numpy())
    reduced_cat_features = pca.fit_transform(combined_cat_feature_maps.view(combined_cat_feature_maps.size(0), -1).cpu().numpy())

    # 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 강아지 클래스 시각화
    scatter_dog = ax.scatter(
        reduced_dog_features[:, 0],
        reduced_dog_features[:, 1],
        reduced_dog_features[:, 2],
        label='Dog',
        alpha=0.5,
    )

    # 고양이 클래스 시각화
    scatter_cat = ax.scatter(
        reduced_cat_features[:, 0],
        reduced_cat_features[:, 1],
        reduced_cat_features[:, 2],
        label='Cat',
        alpha=0.5,
    )

    # 각 축 레이블 설정
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    ax.set_title('PCA 3D Visualization')
    plt.legend()

    # 회전 애니메이션 함수
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return scatter_dog, scatter_cat

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=range(0, 360, 3), blit=True)

    # GIF 파일로 저장
    ani.save('pca_3d_animation.gif', writer='pillow', fps=30)

    # 애니메이션 표시
    plt.show()
        
    
    # 특징 맵 시각화
    #visualize_feature_maps(all_feature_maps)

if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    OUTPUT_DIM = 2
    model = ResNet(config_model(50), OUTPUT_DIM)
    #print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    model_state_dict = torch.load("C:\\Users\\DSEM-Server03\\Desktop\\testdir\\resnet_train_v9.pt", map_location=device)
    # 모델 상태만 가져온거라 모델을  import 해야함!!!! 블로그 참조
    model.load_state_dict(model_state_dict)

    id_list = []
    pred_list = []
    _id=0

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32


    cat_directory = "C:\\Users\\DSEM-Server03\\Desktop\\tmp\\PetImages\\Cat\\"
    dog_directory = "C:\\Users\\DSEM-Server03\\Desktop\\tmp\\PetImages\\Dog\\"

    cat_images_filepaths = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)])   
    dog_images_filepaths = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])
    images_filepaths = [*cat_images_filepaths, *dog_images_filepaths]    
    correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

    #random.seed(42)     
    random.seed(datetime.now().timestamp())
    random.shuffle(correct_images_filepaths)


    test_images_filepaths = correct_images_filepaths[-2000:]    
    print(len(test_images_filepaths))

    label_dic = {'Dog':1, 'Cat':0}

    total = len(test_images_filepaths)
    correct = 0

    phase1 = 'val'
    phase2 = 'random'
    phase3 = 'rotate'
    
    # print(model)

    num_ptrs = model.fc.in_features
    #print("num_ptrs : ", num_ptrs)
    #model.fc = Identity()
    #print(model)

    inference_model(model, size, mean, std, total, phase1)
    # inference_model(model, size, mean, std, total, phase2)
    # inference_model(model, size, mean, std, total, phase3)
