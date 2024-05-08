import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

# 파일 경로
file_path = './data/Wafer-Dataset/Prediction_Batch_files/Learning1.csv'

# pandas를 사용하여 CSV 파일 읽기
data = pd.read_csv(file_path)

# 결측값을 0으로 대체하기
data_filled_zero = data.fillna(0)

# 특성과 타깃 분리
X = data_filled_zero.drop('Sensor-320', axis=1)  # 타깃 센서를 제외한 나머지 센서 데이터
y = data_filled_zero['Sensor-320']  # 예측하고자 하는 센서 데이터

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 상관관계 히트맵 생성
plt.figure(figsize=(20, 15))  # 차트 크기 조정
correlation_matrix = data_filled_zero.corr()  # 상관관계 계산
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()