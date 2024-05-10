import pandas as pd
import os

directory = './data/archive/Training_Batch_Files'  # CSV 파일이 저장된 디렉토리 경로
files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

all_data = pd.DataFrame()  # 모든 데이터를 저장할 빈 데이터프레임 생성

for file in files:
    df = pd.read_csv(file)  # 각 파일을 데이터프레임으로 읽기
    all_data = pd.concat([all_data, df], ignore_index=True)  # 데이터프레임 병합

all_data.to_csv('./data/archive/merged_data2.csv', index=False)  # 인덱스 없이 CSV 파일로 저장
