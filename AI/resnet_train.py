
#6.1.6 ResNset


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image


import matplotlib.pyplot as plt
import numpy as np

import urllib.request
import zipfile

import multiprocessing as mp
from multiprocessing import freeze_support

import copy
from collections import namedtuple
import os
import random
import time

import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError, ImageFile

from ResNet import CBAM, ResNet, BasicBlock, Bottleneck
from ImageTransform import ImageTransform


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (2, 3 두개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)


    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    data_directory = "./Animal/data/train"

    label_paths = sorted([os.path.join(data_directory, f) for f in os.listdir(data_directory)])
    label_names = sorted([f for f in os.listdir(data_directory)])

    image_paths = [sorted([os.path.join(label_path, f) for f in os.listdir(label_path)]) for label_path in label_paths]
    all_image_paths = [path for paths in image_paths for path in paths]
    correct_images_filepaths = [i for i in all_image_paths if cv2.imread(i) is not None]


    random.seed(42)    
    random.shuffle(correct_images_filepaths)
      
    train_images_filepaths = correct_images_filepaths[:int(
        len(correct_images_filepaths) * 0.8)]
    val_images_filepaths = correct_images_filepaths[int(
        len(correct_images_filepaths) * 0.8):int(len(correct_images_filepaths) * 0.9)]
    test_images_filepaths = correct_images_filepaths[int(
        len(correct_images_filepaths) * 0.9):]
    
    print(len(train_images_filepaths), len(val_images_filepaths), len(test_images_filepaths))

    class DogvsCatDataset(Dataset):    
        def __init__(self, file_list, transform=None, phase='train'):    
            self.file_list = file_list
            self.transform = transform
            self.phase = phase
            
        def __len__(self):
            return len(self.file_list)
        
        def __getitem__(self, idx):       
            img_path = self.file_list[idx]
            img = Image.open(img_path).convert('RGB')        
            img_transformed = self.transform(img, self.phase)
            
            label = label_names.index(img_path.split('\\')[-2])
            return img_transformed, label


    train_dataset = DogvsCatDataset(train_images_filepaths, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = DogvsCatDataset(val_images_filepaths, transform=ImageTransform(size, mean, std), phase='val')

    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    train_iterator  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloader_dict = {'train': train_iterator, 'val': valid_iterator}

    batch_iterator = iter(train_iterator)
    inputs, label = next(batch_iterator)
    print('input size : ', inputs.size())
    print('lable : ', label)
    print('type :', type(train_iterator))

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])


    resnet18_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [2,2,2,2],
                                channels = [64, 128, 256, 512])

    resnet34_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [3,4,6,3],
                                channels = [64, 128, 256, 512])

    resnet50_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 6, 3],
                                channels = [64, 128, 256, 512])

    resnet101_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 23, 3],
                                    channels = [64, 128, 256, 512])

    resnet152_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 8, 36, 3],
                                    channels = [64, 128, 256, 512])


    OUTPUT_DIM = len(label_names)
    model = ResNet(resnet50_config, OUTPUT_DIM)
    #print(model)


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.layer3[-1].add_module('cbam', CBAM(1024))
    # # Pretrained ResNet 모델 불러오기
    # pretrained_model = models.resnet50(pretrained=True)
    # pretrained_dict = pretrained_model.state_dict() 

    # # 현재 모델의 state_dict
    # model_dict = model.state_dict()

    # # Pretrained dict에서 fc layer 및 불필요한 key 제거
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}

    # # 기존 모델 dict에 pretrained dict 병합
    # model_dict.update(pretrained_dict) 

    # # 업데이트된 state_dict 모델에 로드
    # model.load_state_dict(model_dict)

    criterion = criterion.to(device)



    def calculate_topk_accuracy(y_pred, y, k = 2):
        with torch.no_grad():
            batch_size = y.shape[0]
            _, top_pred = y_pred.topk(k, 1)
            top_pred = top_pred.t()
            correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
            correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            acc_1 = correct_1 / batch_size
            acc_k = correct_k / batch_size
        return acc_1, acc_k


    def train(model, iterator, optimizer, criterion, device):    
        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0
        
        model.train()    
        for (x, y) in iterator:        
            x = x.to(device)
            y = y.to(device)
                
            optimizer.zero_grad()                
            y_pred = model(x)  
            
            loss = criterion(y_pred[0], y) 
            
            acc_1, acc_5 = calculate_topk_accuracy(y_pred[0], y)        
            loss.backward()        
            optimizer.step()        
            
            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
            
        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)        
        return epoch_loss, epoch_acc_1, epoch_acc_5

    def evaluate(model, iterator, criterion, device):    
        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0
        
        model.eval()    
        with torch.no_grad():        
            for (x, y) in iterator:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)            
                loss = criterion(y_pred[0], y)
                acc_1, acc_5 = calculate_topk_accuracy(y_pred[0], y)

                epoch_loss += loss.item()
                epoch_acc_1 += acc_1.item()
                epoch_acc_5 += acc_5.item()
            
        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)        
        return epoch_loss, epoch_acc_1, epoch_acc_5

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    import time
    import wandb

    best_valid_loss = float('inf')
    EPOCHS = 3000
    early_stopping_epochs = 3
    early_stop_counter = 0

    # CBAM 추가 테스트
    # model.layer3[-1].add_module('cbam', CBAM(1024))

    wandb.init(project='resnet_fine-tuning_experiment')
    wandb.run.name = 'resnet-cbam04-AMF'
    print(wandb.run.name)
    wandb.run.save()

    args = {
        "epochs": EPOCHS,
        "batch_size": batch_size
    }
    wandb.config.update(args)

    for epoch in range(EPOCHS):    
        start_time = time.monotonic()
        
        train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{wandb.run.name}.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
            f'Train Acc @5: {train_acc_5*100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
            f'Valid Acc @5: {valid_acc_5*100:6.2f}%')
        wandb.log({"Training loss": train_loss,
                "Valid loss": valid_loss,
                "Training acc": train_acc_1,
                "Valid acc": valid_acc_1})

        if early_stop_counter >= early_stopping_epochs:
            print("!!!!!Early Stopping!!!!!")
            break

    import pandas as pd
    id_list = []
    pred_list = []
    _id=0

    total = len(test_images_filepaths)
    correct = 0

    with torch.no_grad():
        for test_path in test_images_filepaths:
            img = Image.open(test_path).convert('RGB')
            _id =test_path.split('\\')[-1].split('.')[0]
            label_index = label_names.index(test_path.split('\\')[-2])
            
            transform = ImageTransform(size, mean, std)
            img = transform(img, phase='val')
            img = img.unsqueeze(0)
            img = img.to(device)

            model.eval()
            outputs = model(img)[0]
            preds_label_index = F.softmax(outputs, dim=1)[:, 1].tolist()
            
            if preds_label_index == label_index:
                correct += 1

            id_list.append(_id)
            pred_list.append(preds_label_index)

    print(f"Test data ACC : {correct} / {total} = {correct / total}")


    # def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
        
    #     res = pd.DataFrame({
    #     'id': id_list,
    #     'label': pred_list
    #     })

    #     res.sort_values(by='id', inplace=True)
    #     res.reset_index(drop=True, inplace=True)

    #     res.to_csv('C:\\Users\\DSEM-Server03\\Desktop\\testdir\\test_v02.csv', index=False)
    #     res.head(10)

    #     class_ = classes = {0:'cat', 1:'dog'}
    #     rows = len(images_filepaths) // cols

    #     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    #     for i, image_filepath in enumerate(images_filepaths):
    #         image = cv2.imread(image_filepath)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    #         a = random.choice(res['id'].values)    
    #         label = res.loc[res['id'] == a, 'label'].values[0]
            
    #         if label > 0.5:
    #             label = 1
    #         else:
    #             label = 0

    #         ax.ravel()[i].imshow(image)
    #         ax.ravel()[i].set_title(class_[label])
    #         ax.ravel()[i].set_axis_off()
    #     plt.tight_layout()
    #     plt.show()

    # display_image_grid(test_images_filepaths) 
    # print(test_images_filepaths[0])
    # img = cv2.imread(test_images_filepaths[0])
    #bplt.imshow(img)
    # plt.show()
