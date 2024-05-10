import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_rows', None)  # 모든 행 출력
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', None)  # 셸 너비에 맞춰 출력
pd.set_option('display.max_colwidth', None)  # 열의 최대 너비 제한 없앰

# 데이터 불러오기
data = pd.read_csv('./data/Wafer-Dataset/Prediction_Batch_files/Test_Triple.csv')

# 전체가 nan인 열 제거
data = data.dropna(axis=1, how='all')

# NaN이 아닌 데이터가 있는 열만 평균을 계산
valid_means = data.mean(skipna=True)

# 결측치 대체
data.fillna(valid_means, inplace=True)

