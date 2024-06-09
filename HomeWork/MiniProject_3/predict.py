import os
import glob
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# 클래스 이름 정의
class_names = ['battery', 'bottle_brown', 'bottle_transparent', 'can_aluminium', 'colored_pet', 'light']

# 모델 로드
model_path = './data/epoch_15_class_6/model.pth'
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(model_path))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return class_names[predicted.item()], image

def plot_results(image_paths, save_dir):
    num_images = len(image_paths)
    num_plots = (num_images + 3) // 4  # 한 플롯에 최대 4개의 이미지를 배치
    os.makedirs(save_dir, exist_ok=True)

    for plot_idx in range(num_plots):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        plot_image_paths = image_paths[plot_idx*4:(plot_idx+1)*4]

        for i, image_path in enumerate(plot_image_paths):
            pred_class, image = predict_image(image_path)
            ax = axes[i]
            ax.imshow(image)
            ax.set_title(f"Predicted: {pred_class}")
            ax.axis('off')

        # 나머지 플롯 숨기기
        for j in range(len(plot_image_paths), 4):
            ax = axes[j]
            ax.axis('off')

        plt.subplots_adjust(top=0.85)
        plot_filename = f'plot_{plot_idx + 1}.png'
        plt.savefig(os.path.join(save_dir, plot_filename))
        plt.close()

if __name__ == "__main__":
    # 이미지 파일 경로 리스트
    image_dir = './data/predict_test'
    image_paths = glob.glob(os.path.join(image_dir, '*'))

    # 결과 플롯 저장 디렉토리
    save_dir = './data/predict_test_plots'

    # 결과 플롯
    plot_results(image_paths, save_dir)
