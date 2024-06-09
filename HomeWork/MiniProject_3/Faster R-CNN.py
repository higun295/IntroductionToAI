import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import json

# 모델 로드
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 이미지 변환
transform = transforms.Compose([
    transforms.ToTensor()
])

# 이미지 디렉토리 및 저장 경로 설정
image_dir = './data'  # 여기에 이미지를 저장한 디렉토리 경로를 입력하세요
output_dir = './Faster R-CNN Result'  # 라벨링된 이미지를 저장할 디렉토리 경로를 입력하세요
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 객체 탐지 및 라벨링 결과 저장 및 시각화
for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)
    image = Image.open(img_path)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    labels = []
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score >= 0.5:  # 신뢰도가 50% 이상인 경우만 사용
            labels.append({
                'class': label.item(),
                'x1': box[0].item(),
                'y1': box[1].item(),
                'x2': box[2].item(),
                'y2': box[3].item(),
                'confidence': score.item()
            })

            # 시각화
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), f"{label.item()} {score:.2f}", fill="red", font=font)

    # 결과 JSON 파일로 저장
    output_path = os.path.join(output_dir, image_file.replace('.jpg', '.json').replace('.png', '.json'))
    with open(output_path, 'w') as f:
        json.dump(labels, f)

    # 라벨링된 이미지 저장
    labeled_image_path = os.path.join(output_dir, image_file)
    image.save(labeled_image_path)

print("라벨링 작업 완료!")
