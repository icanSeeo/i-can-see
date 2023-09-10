import torch
from utils import *
from model import *
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 추론을 수행하는 함수 정의
def perform_inference(model, input_batch):
    with torch.no_grad():
        output, attention_map = model(input_batch)
    return output, attention_map

# 결과를 해석하는 함수 정의
def interpret_output(output, class_names):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(class_names[top5_catid[i]], top5_prob[i].item())

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Resize and crop images to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and standard deviation
])

model = ResNetWithCBAM(Bottleneck, [3, 4, 6, 3], num_classes=1000)
model.apply_pretrained()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 300
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Load the CIFAR-10 dataset
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)

# Define a DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in tqdm(range(num_epochs)):
    model.train()

    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, aux_output = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    

model.save_weights('cbam-finetunned-300-notpretrained.pth')