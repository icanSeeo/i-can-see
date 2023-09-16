import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import json

# Load class labels from JSON file
with open('imagenet_classes.json') as f:
    class_names = json.load(f)

# Load pre-trained ResNet model
model = resnet50(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


if __name__ == "__main__":
    # Load and preprocess the image
    image_path = 'piano.jpg'
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Use GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        input_batch = input_batch.to('cuda')

    # Inference
    with torch.no_grad():
        output = model(input_batch)

    # Convert output to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Print top 5 predictions
    for i in range(top5_prob.size(0)):
        print(class_names[top5_catid[i]], top5_prob[i].item())
