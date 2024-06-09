import os
import pandas as pd

def merge_csv_files(directory, output_file):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    all_dataframes = []

    for file in all_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df['source_file'] = file  # 파일명을 새로운 열로 추가
        all_dataframes.append(df)

    # 모든 데이터프레임을 하나로 합침
    merged_dataframe = pd.concat(all_dataframes, ignore_index=True)

    # 합친 데이터를 새로운 CSV 파일로 저장
    merged_dataframe.to_csv(output_file, index=False)
    print(f'Merged CSV saved to {output_file}')

# 디렉토리 경로와 출력 파일 경로 설정
directory = './data/hyperparameter_test/Learning_Rate_0.001'
output_file = './data/hyperparameter_test/merged_learning_rate_0.001.csv'

# CSV 파일 병합 함수 호출
merge_csv_files(directory, output_file)
