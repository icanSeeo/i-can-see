import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import urllib.request
import zipfile
import glob
import os
from PIL import Image, UnidentifiedImageError, ImageFile

import multiprocessing as mp
from multiprocessing import freeze_support

import matplotlib.pyplot as plt


class ImageAug():
    def imtransform(self, opt, transform):
        batch_size = 16
        original_images = self.originalset()
        # 이미지 폴더로부터 데이터를 로드합니다.
        transform_dataset = ImageFolder(root='C:\\Users\\DSEM-Server03\\Desktop\\tmp\\PetImages',  # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                        transform=transform)

        # 데이터 로더를 생성합니다.
        transform_loader = DataLoader(transform_dataset,  # 이전에 생성한 transform_dataset를 적용합니다.
                                      batch_size=batch_size,  # 배치사이즈
                                      shuffle=False,  # 셔플 여부
                                      num_workers=1
                                      )

        transform_images, labels = next(iter(transform_loader))

        fig, axes = plt.subplots(3, 2)
        fig.set_size_inches(4, 6)

        if opt == "show":
            for idx in range(3):
                axes[idx, 0].imshow(original_images[idx].permute(1, 2, 0))
                axes[idx, 0].set_axis_off()
                axes[idx, 0].set_title('Original')
                axes[idx, 1].imshow(transform_images[idx].permute(1, 2, 0))
                axes[idx, 1].set_axis_off()
                axes[idx, 1].set_title('Transformed')
            fig.tight_layout()
            plt.show()
        elif opt == "save":
            os.makedirs('tmp', exist_ok=True)
            # 폴더 생성 및 이미지 저장
            for idx in range(batch_size):
                save_image(transform_images[idx], 'tmp/test'+str(idx)+'.png')
        elif opt == "get_tensor":
            return transform_images
        else:
            raise Exception("not correct command")

        print(type(transform_images[0]))
        print(type(transform_images))
        print(type(transform_dataset))

    def originalset(self):
        # 랜덤 시드 설정
        torch.manual_seed(321)
        # 이미지 크기를 224 x 224 로 조정합니다
        IMAGE_SIZE = 224

        original_dataset = ImageFolder(root='C:\\Users\\DSEM-Server03\\Desktop\\tmp\\PetImages',  # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                       transform=transforms.Compose([  # Resize 후 정규화(0~1)를 수행합니다.
                                           transforms.Resize(
                                               (IMAGE_SIZE, IMAGE_SIZE)),
                                           # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
                                           transforms.ToTensor()
                                       ]))

        original_loader = DataLoader(original_dataset,  # 이전에 생성한 original_dataset를 로드 합니다.
                                     batch_size=16,  # 배치사이즈
                                     shuffle=False,  # 셔플 여부
                                     num_workers=1
                                     )

        # 1개의 배치를 추출합니다.
        original_images, labels = next(iter(original_loader))

        # 이미지의 shape을 확인합니다. 224 X 224 RGB 이미지 임을 확인합니다.
        # (batch_size, channel, height, width)
        original_images.shape
        return original_images

    def trans_colorJitter(self, brightness, contrast, saturation, hue):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # ColorJitter 적용                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            transforms.ColorJitter(brightness=brightness,
                                   contrast=contrast,
                                   saturation=saturation,
                                   hue=hue,
                                   ),
            transforms.ToTensor()
        ])
        return image_transform

    def trans_horizontal_flip(self, p):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # RandomHorizontalFlip 적용
            transforms.RandomHorizontalFlip(p=p),
            transforms.ToTensor()
        ])
        return image_transform

    def trans_gaussian_blur(self, kernel_size, sigma):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # GaussianBlur 적용
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            transforms.ToTensor()
        ])
        return image_transform

    def trans_rotate(self, degrees, fill):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # RandomRotation 적용
            transforms.RandomRotation(
                degrees=degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill=fill),
            transforms.ToTensor()
        ])
        return image_transform

    def trans_padding(self, padding, fill, padding_mode):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # Pad 적용
            transforms.Pad(padding=padding, fill=fill,
                           padding_mode=padding_mode),
            transforms.ToTensor()
        ])
        return image_transform

    def autotransform(self, policy):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # AutoAugment 적용
            transforms.AutoAugment(policy=policy,
                                   interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        return image_transform

    def autotransform(self, policy):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # AutoAugment 적용
            transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,
                                   interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        return image_transform


if __name__ == '__main__':
    freeze_support()
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    IA = ImageAug()
    # IA.download_dataset(
    #    'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip',
    #    'PetImages') # 데이터셋 zip 경로, 생성할 폴더 이름
    # transform_info = IA.trans_colorJitter((0.5, 0.9), (0.4, 0.8), (0.7, 0.9), (-0.2, 0.2))
    # transform_info = IA.trans_horizontal_flip(0.8)
    transform_info = IA.trans_gaussian_blur((19, 19), (1.0, 2.0))
    #transform_info = IA.trans_rotate((-30, 30), 0)
    #transform_info = IA.trans_padding((100, 50, 100, 200), 255, 'symmetric')
    # transform_info = IA.autotransform(
    # transforms.autoaugment.AutoAugmentPolicy.IMAGENET)

    IA.imtransform("show", transform_info)
