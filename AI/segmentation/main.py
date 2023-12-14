import yolov5

print(yolov5)

import torch
from models.yolo import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(cfg='models/yolov5n.yaml', ch=3, nc=80).to(device)

# Print the model summary with layer names and output shapes
print("Layer (type)\t\t\t\tOutput Shape\t\t\tParam #")
print("==================================================================================")
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        total_params += num_params
        if param.dim() == 1:
            print(f"{name:<40}{param.size()}\t\t{num_params}")
        else:
            print(f"{name:<40}{list(param.size())}\t\t\t{num_params}")

print("==================================================================================")
print(f"Total params: {total_params}")