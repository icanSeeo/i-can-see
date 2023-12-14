import torch
from utils import *
from model import *
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# Define functions for inference and interpretation


def perform_inference(model, input_batch):
    with torch.no_grad():
        output, attention_map = model(input_batch)
    return output, attention_map


def interpret_output(output, class_names):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(class_names[top5_catid[i]], top5_prob[i].item())


# Define transformations and load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = ResNetWithCBAM(Bottleneck, [3, 4, 6, 3], num_classes=1000)
model.apply_pretrained()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 200
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Load the CIFAR-100 dataset
train_dataset = CIFAR100(root='./data', train=True,
                         download=True, transform=transform)

# Define a DataLoader
batch_size = 200
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize lists to store accuracy and loss
accuracy_list = []
loss_list = []

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

    # Calculate accuracy after each epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, aux_output = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracy_list.append(accuracy)
    loss_list.append(running_loss / len(train_loader))

    print(
        f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")

# Plot the accuracy graph
plt.plot(range(1, num_epochs+1), accuracy_list, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

# Plot the loss graph
plt.plot(range(1, num_epochs+1), loss_list, label='Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Save the model weights
model.save_weights('cbam-finetunned-200-notpretrained(200batch).pth')
