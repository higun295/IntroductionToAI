import torch
import os
import json
from pathlib import Path
from PIL import Image

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 사전 학습된 모델 로드

# 이미지 디렉토리 및 저장 경로 설정
image_dir = './data'
output_dir = './labeled_data'
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 객체 탐지 및 라벨링 결과 저장
for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)
    results = model(img_path)

    # 결과 JSON 파일로 저장
    labels = []
    for det in results.xyxy[0]:  # x1, y1, x2, y2, conf, class
        label = {
            'class': int(det[5]),
            'x1': float(det[0]),
            'y1': float(det[1]),
            'x2': float(det[2]),
            'y2': float(det[3]),
            'confidence': float(det[4])
        }
        labels.append(label)

    output_path = os.path.join(output_dir, image_file.replace('.jpg', '.json').replace('.png', '.json'))
    with open(output_path, 'w') as f:
        json.dump(labels, f)

    # 라벨링된 이미지 시각화 및 저장 (선택 사항)
    results.save(save_dir=output_dir)

print("라벨링 작업 완료!")
