import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
from glob import glob
import wandb

dataset_base_uri = '/home/jun0zo/Experiments/i-can-see/AI/object_detection/datasets'

def train():

    wandb.init(project='yolov8_fine-tuning_experiment', entity='joon0zo1022')
    
    
    wandb.run.name = 'YOLOv8_baseline_RS'
    print(wandb.run.name)
    wandb.run.save()

    args = {
        "epochs": 250
    }
    wandb.config.update(args)

    img_list = glob(f'{dataset_base_uri}/train/images/*.jpg')
    val_img_list = glob(f'{dataset_base_uri}/valid/images/*.jpg')
    test_img_list = glob(f'{dataset_base_uri}/test/images/*.jpg')
    
    with open(os.path.join(dataset_base_uri, 'train.txt'), 'w') as f:
      f.write('\n'.join(img_list) + '\n')
        
    with open(os.path.join(dataset_base_uri, 'valid.txt'), 'w') as f:
      f.write('\n'.join(val_img_list) + '\n')
    
    with open(os.path.join(dataset_base_uri, 'test.txt'), 'w') as f:
      f.write('\n'.join(test_img_list) + '\n')
        
    model = YOLO('yolov8n.pt')
    wandb.watch(model)
    
    model.train(data=os.path.join(dataset_base_uri, 'data.yaml'), epochs=250)

    print('Done!')

def inference():
    model = YOLO('./runs/detect/train13/weights/best.pt')
    
    for (root, dirs, files) in os.walk(f"{dataset_base_uri}/test/images"):
        for file in files:
            result = model.predict(f"{dataset_base_uri}/test/images/{file}", save=True)

    # plots = results[0].plot()
    # img = cv2.resize(plots, (600, 800))
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    # train()
    inference()