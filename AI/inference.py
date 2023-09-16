import torch
from utils import *
from model import *

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


# 이미지 경로 설정
image_path = 'data/cifar-100-python/train/buildings'
# 이미지 전처리 및 모델 준비
input_batch = preprocess_image(image_path)

model = ResNetWithCBAM.load_weights(
    'cbam-finetunned.pth', Bottleneck, [3, 4, 6, 3], num_classes=1000)

model.eval()
model, input_batch = prepare_model_and_input(model, input_batch)

# 추론 수행
output, attention_map = perform_inference(model, input_batch)

# 클래스 인덱스를 클래스명으로 변환
class_names = load_class_names()

# 결과 해석 및 출력
interpret_output(output, class_names)

original_image = cv2.imread('piano.jpg')
print(original_image.shape)
overlay_attention_map(
    original_image, attention_map.cpu().numpy(), 'output_image.jpg')
