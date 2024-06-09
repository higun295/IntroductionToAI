import os
import shutil
import random

# 데이터셋 디렉토리 설정
dataset_dir = 'path_to_labeled_images'
train_dir = 'path_to_train'
val_dir = 'path_to_val'
test_dir = 'path_to_test'

# 디렉토리 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 파일 리스트 가져오기
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 데이터 분할 (예: 70% 학습, 20% 검증, 10% 테스트)
random.shuffle(image_files)
train_split = int(0.7 * len(image_files))
val_split = int(0.9 * len(image_files))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# 파일 이동
for file in train_files:
    shutil.copy(os.path.join(dataset_dir, file), train_dir)

for file in val_files:
    shutil.copy(os.path.join(dataset_dir, file), val_dir)

for file in test_files:
    shutil.copy(os.path.join(dataset_dir, file), test_dir)

print("데이터 분할 완료!")
