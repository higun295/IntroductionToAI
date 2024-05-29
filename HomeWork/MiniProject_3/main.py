import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# JSON 파일 로드
def load_via_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# 커스텀 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        self.data = load_via_json(json_path)
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = []
        for key, value in self.data.items():
            file_name = value['filename']
            regions = value['regions']
            for region in regions:
                shape_attributes = region['shape_attributes']
                x = shape_attributes['x']
                y = shape_attributes['y']
                width = shape_attributes['width']
                height = shape_attributes['height']
                label = region['region_attributes']['Label']
                self.annotations.append((file_name, x, y, width, height, label))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        file_name, x, y, width, height, label = self.annotations[idx]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        box = torch.tensor([x, y, x + width, y + height], dtype=torch.float32)
        return image, box, label


# 데이터 변환
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 및 데이터로더 정의
json_path = 'data/your_via_project.json'  # 실제 JSON 파일 경로로 변경
img_dir = 'data/images'  # 실제 이미지 디렉토리 경로로 변경
dataset = CustomDataset(json_path, img_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# 샘플 데이터 확인
def show_sample(sample):
    image, box, label = sample
    image = image.permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5)

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    plt.title(label)
    plt.show()


# 샘플 데이터 출력
sample = next(iter(dataloader))
show_sample(sample)
