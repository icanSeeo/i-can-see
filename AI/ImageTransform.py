import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms

# 사용자 정의 노이즈 추가 함수들
def add_dust(img_tensor, dust_density=0.01):
    dust_mask = torch.rand_like(img_tensor) < dust_density
    dust = torch.randn_like(img_tensor) * 0.5  # 조절 가능한 dust 강도
    img_tensor[dust_mask] += dust[dust_mask]
    return torch.clamp(img_tensor, 0, 1)

def add_lighting(img_tensor, light_density=1.2):
    lighting = img_tensor * light_density
    return lighting

def add_rotation(img_tensor, max_angle=10):
    angle = np.random.uniform(-max_angle, max_angle)
    img_pil = transforms.ToPILImage()(img_tensor)
    rotated_img_pil = img_pil.rotate(angle)
    return transforms.ToTensor()(rotated_img_pil)

def add_blur(img_tensor, blur_radius=2):
    img_pil = transforms.ToPILImage()(img_tensor)
    blurred_img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return transforms.ToTensor()(blurred_img_pil)

# rescale value to [min, max]
def rescale(value, min, max):
    return min + (max - min) * value

class ImageTransform():    
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), noise_weight=0.0):
        self.norm_mean = mean
        self.norm_std = std
        
        self.light_density = rescale(noise_weight, 1.0, 3.0)
        self.dust_density = rescale(noise_weight, 0.0, 2.0)
        self.rotate_angle = rescale(noise_weight, 0.0, 180.0)
        self.blur_radius = rescale(noise_weight, 0.0, 10.0)
        
        
        print('light_density :', self.light_density)
        print('dust_density :', self.dust_density)
        print('rotate_angle :', self.rotate_angle)
        print('blur_radius :', self.blur_radius)
        
        
        
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            
            'light': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_lighting(x, light_density=self.light_density)),
                transforms.Normalize(mean, std),
            ]),
            'dust': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_dust(x, dust_density=self.dust_density)),
                transforms.Normalize(mean, std)
                
            ]),
            'rotate': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.RandomRotation((self.rotate_angle, self.rotate_angle+1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'blur': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_blur(x, blur_radius=self.blur_radius)),
                transforms.Normalize(mean, std),
                
            ]),
            # 'colorJit': transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(resize),
            #     transforms.ToTensor(),
            #     transforms.ColorJitter(brightness=(0.9, 0.9),
            #                        contrast=(0.4, 0.4),
            #                        saturation=(0.7, 0.7),
            #                        hue=(-0.2, 0.2),
            #                        ),
            #     transforms.Normalize(mean, std),
                
            # ]),
            
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)
    
    def denormalize(self, normalized_image):
        # Transpose (C, H, W) to (H, W, C)
        denormalized_image = normalized_image.permute(1, 2, 0).numpy()

        # Denormalize each channel
        for i in range(3):
            denormalized_image[:, :, i] = denormalized_image[:, :, i] * self.norm_std[i] + self.norm_mean[i]

        # Clip values to be in [0, 1] range
        denormalized_image = np.clip(denormalized_image, 0, 1)

        # Convert to uint8 if necessary
        denormalized_image = (denormalized_image * 255).astype(np.uint8)

        return denormalized_image