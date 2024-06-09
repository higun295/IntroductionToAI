import os
import itertools
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 설정 값
learning_rates = [0.001, 0.01, 0.0001]
batch_sizes = [16, 32, 64]
num_epochs_list = [5, 10, 15]
resize_sizes = [(224, 224), (299, 299), (128, 128)]

# 결과 저장 폴더 생성
results_dir = './data/hyperparameter_test'
os.makedirs(results_dir, exist_ok=True)

# 데이터셋 경로
train_path = 'D:/pytorch_images2/Training'
val_path = 'D:/pytorch_images2/Validation'

# 데이터 샘플링 비율 (예: 5%의 데이터 사용)
sample_ratio = 0.05

def get_sampled_dataset(dataset, sample_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    sample_size = int(np.floor(sample_ratio * dataset_size))
    sampled_indices = indices[:sample_size]
    return Subset(dataset, sampled_indices)

def train_and_evaluate(lr, batch_size, num_epochs, resize_size):
    # 데이터 전처리 정의
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 및 데이터로더 정의
    full_train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    full_val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    # 데이터 샘플링
    train_dataset = get_sampled_dataset(full_train_dataset, sample_ratio)
    val_dataset = get_sampled_dataset(full_val_dataset, sample_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 정의
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)  # 클래스 수에 맞게 출력 레이어 수정

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습 및 검증 루프
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", leave=False)
        for inputs, labels in train_loader_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            train_loader_iter.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_accuracy = corrects.double() / total

        # 검증 단계
        model.eval()
        val_loss = 0.0
        corrects = 0
        total = 0
        val_loader_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loader_iter:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                val_loader_iter.set_postfix(loss=loss.item())

        val_loss = val_loss / total
        val_accuracy = corrects.double() / total

        # 로그 데이터 저장
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy.item())
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy.item())

    return history

if __name__ == "__main__":
    # 모든 조합에 대해 실험 수행
    total_combinations = len(learning_rates) * len(batch_sizes) * len(num_epochs_list) * len(resize_sizes)
    current_combination = 0

    for lr, batch_size, num_epochs, resize_size in tqdm(
            itertools.product(learning_rates, batch_sizes, num_epochs_list, resize_sizes), total=total_combinations,
            desc="Total Progress"):
        current_combination += 1
        print(
            f'Starting training with LR: {lr}, Batch Size: {batch_size}, Epochs: {num_epochs}, Resize: {resize_size} ({current_combination}/{total_combinations})')
        history = train_and_evaluate(lr, batch_size, num_epochs, resize_size)

        # 로그 데이터프레임으로 변환 및 저장
        log_df = pd.DataFrame(history)
        log_filename = f'log_lr_{lr}_bs_{batch_size}_epochs_{num_epochs}_resize_{resize_size[0]}x{resize_size[1]}.csv'
        log_df.to_csv(os.path.join(results_dir, log_filename), index=False)

        # 그래프 그리기
        fig, ax = plt.subplots(2, 1, figsize=(12, 16))
        ax[0].plot(history['epoch'], history['train_loss'], label='Training Loss')
        ax[0].plot(history['epoch'], history['val_loss'], label='Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training and Validation Loss')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(history['epoch'], history['train_accuracy'], label='Training Accuracy')
        ax[1].plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_title('Training and Validation Accuracy')
        ax[1].legend()
        ax[1].grid(True)

        plot_filename = f'plot_lr_{lr}_bs_{batch_size}_epochs_{num_epochs}_resize_{resize_size[0]}x{resize_size[1]}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, plot_filename))
        plt.close()

        print(
            f'Finished training with LR: {lr}, Batch Size: {batch_size}, Epochs: {num_epochs}, Resize: {resize_size}, Logs and plots saved.')

