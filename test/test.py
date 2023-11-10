import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def test_training():
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # ResNet 모델 정의
    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet.to(device)

    # 손실 함수 및 최적화 함수
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.01)

    # 모델 학습
    num_epochs = 5

    for epoch in range(num_epochs):
        running_loss = 0.0
        resnet.train()

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

 
    correct = 0
    total = 0
    resnet.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

 
    assert correct / total >= 0.9, f"Accuracy: {100 * correct / total}%"
