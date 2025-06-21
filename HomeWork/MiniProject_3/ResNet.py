import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # 디렉토리 생성
    os.makedirs('./data/epoch_15_class_6', exist_ok=True)

    # 모델 정의 및 초기화
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)  # 7개의 클래스를 출력으로 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 데이터셋 경로 및 전처리 정의
    train_path = 'D:/images/Training'
    val_path = 'D:/images/Validation'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 학습 파라미터 설정
    num_epochs = 15

    # 로그 데이터 저장을 위한 리스트 초기화
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
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", unit="batch"):
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

        train_loss = running_loss / total
        train_accuracy = corrects.double() / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        # 검증 단계
        model.eval()
        val_loss = 0.0
        corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_loss = val_loss / total
        val_accuracy = corrects.double() / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # 로그 데이터 저장
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy.item())
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy.item())

        # 로그 데이터프레임으로 변환 및 저장
        history_df = pd.DataFrame(history)
        history_df.to_csv('./data/epoch_15_class_6/training_log.csv', index=False)

        # 그래프 그리기
        plt.figure(figsize=(12, 8))
        plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
        plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
        plt.plot(history['epoch'], history['train_accuracy'], label='Training Accuracy')
        plt.plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Accuracy')
        plt.title('Training and Validation Loss/Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./data/epoch_15_class_6/epoch_{epoch + 1}.png')
        plt.close()

    # 최종 모델 저장
    torch.save(model.state_dict(), './data/epoch_15_class_6/model.pth')
    print("Model saved to './data/epoch_15_class_6/model.pth'")

if __name__ == "__main__":
    main()
