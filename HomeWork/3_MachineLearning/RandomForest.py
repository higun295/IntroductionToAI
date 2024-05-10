import pandas as pd
import numpy as np
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

# ================================================================================

sensor_name = 'Sensor-1'

# 독립 변수와 종속 변수 설정
X = data.drop(sensor_name, axis=1)
y = data[sensor_name]

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=15)
model.fit(X_train, y_train)

# 결정계수
relation_square = model.score(X_train, y_train)

# 피처 중요도 추출
importances = model.feature_importances_

# 중요도에 따라 피처의 인덱스를 정렬
indices = np.argsort(importances)[::-1]

# 상위 100개 피처 선택
top_100_indices = indices[:100]
top_100_features = X.columns[top_100_indices]

plt.figure(figsize=(10, 15))
plt.title('Top 100 Feature Importances of {}'.format(sensor_name))
plt.barh(range(len(top_100_indices)), importances[top_100_indices], color='b', align='center')
plt.yticks(range(len(top_100_indices)), [X.columns[i] for i in top_100_indices])
plt.gca().invert_yaxis()  # 높은 중요도를 가진 피처가 위로 오도록 y축을 역순으로 설정
plt.xlabel('Relative Importance')
plt.show()
