import torchvision.transforms as transforms
class ImageTransform():    
    def __init__(self, resize, mean, std):
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
            'gaussian': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.GaussianBlur(kernel_size=(19, 19), sigma=(1.0, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'colorJit': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ColorJitter(brightness=(0.5, 0.9),
                                   contrast=(0.4, 0.8),
                                   saturation=(0.7, 0.9),
                                   hue=(-0.2, 0.2),
                                   ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'flip': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.RandomHorizontalFlip(p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'padding': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.Pad((100,50,100,200), 255, 'symmetric'),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'rotate': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.RandomRotation((-30, 30), 0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'random':transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,
                                   interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)