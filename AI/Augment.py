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
    def run(self, transform):
        original_images = self.originalset()
        # 이미지 폴더로부터 데이터를 로드합니다.
        transform_dataset = ImageFolder(root='tmp/PetImages',  # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                        transform=transform)

        # 데이터 로더를 생성합니다.
        transform_loader = DataLoader(transform_dataset,  # 이전에 생성한 transform_dataset를 적용합니다.
                                      batch_size=32,  # 배치사이즈
                                      shuffle=False,  # 셔플 여부
                                      num_workers=1
                                      )

        transform_images, labels = next(iter(transform_loader))

        os.makedirs('tmp', exist_ok=True)

        for idx in range(32):
            save_image(transform_images[idx], 'tmp/test'+str(idx)+'.png')


#################### 이미지 데이터 셋 다운로드 및 검증 #########################
    # 이미지 Validation을 수행하고 Validate 여부를 return 합니다.
    def validate_image(self, filepath):
        # image extensions
        image_extensions = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp', '.png', '.avif', '.gif']
        # 이미지 파일 확장자를 가진 경우
        if any(filepath.endswith(s) for s in image_extensions):
            try:
                # PIL.Image로 이미지 데이터를 로드하려고 시도합니다.
                img = Image.open(filepath).convert('RGB')
                img.load()
            except UnidentifiedImageError:  # corrupt 된 이미지는 해당 에러를 출력합니다.
                print(f'Corrupted Image is found at: {filepath}')
                return False
            except (IOError, OSError):  # Truncated (잘린) 이미지에 대한 에러를 출력합니다.
                print(f'Truncated Image is found at: {filepath}')
                return False
            else:
                return True
        else:
            print(f'File is not an image: {filepath}')
            return False

    def download_dataset(self, download_url, folder, default_folder='tmp'):
        # 데이터셋을 다운로드 합니다.
        urllib.request.urlretrieve(download_url, 'datasets.zip')

        # 다운로드 후 tmp 폴더에 압축을 해제 합니다.
        local_zip = 'datasets.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall(f'{default_folder}/')
        zip_ref.close()

        # 잘린 이미지 Load 시 경고 출력 안함
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # image 데이터셋 root 폴더
        root = f'{default_folder}/{folder}'

        dirs = os.listdir(root)

        for dir_ in dirs:
            folder_path = os.path.join(root, dir_)
            files = os.listdir(folder_path)

            images = [os.path.join(folder_path, f) for f in files]
            for img in images:
                valid = self.validate_image(img)
                if not valid:
                    # corrupted 된 이미지 제거
                    os.remove(img)

        folders = glob.glob(f'{default_folder}/{folder}/*')
        print(folders)
        return folders
########################################################################

    def originalset(self):
        # 랜덤 시드 설정
        torch.manual_seed(321)
        # 이미지 크기를 224 x 224 로 조정합니다
        IMAGE_SIZE = 224

        original_dataset = ImageFolder(root='tmp/PetImages',  # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                       transform=transforms.Compose([  # Resize 후 정규화(0~1)를 수행합니다.
                                           transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
            transforms.RandomRotation(degrees=degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill=fill),
            transforms.ToTensor()
        ])
        return image_transform

    def trans_padding(self, padding, fill, padding_mode):
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # Pad 적용
            transforms.Pad(padding=padding, fill=fill, padding_mode=padding_mode),
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

    IA = ImageAug()
    # IA.download_dataset(
    #    'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip',
    #    'PetImages') # 데이터셋 zip 경로, 생성할 폴더 이름
    # transform_info = IA.trans_colorJitter((0.5, 0.9), (0.4, 0.8), (0.7, 0.9), (-0.2, 0.2))
    # transform_info = IA.trans_horizontal_flip(0.8)
    # transform_info = IA.trans_gaussian_blur((19, 19), (1.0, 2.0))
    # transform_info = IA.trans_rotate((-30, 30), 0)
    # transform_info = IA.trans_padding((100, 50, 100, 200), 255, 'symmentric')
    transform_info = IA.autotransform(transforms.autoaugment.AutoAugmentPolicy.IMAGENET)


    IA.run(transform_info)