from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./data/Wafer-Dataset/Prediction_Batch_files/Learning1.csv')

# 결측값을 0으로 대체하기
data_filled_zero = data.fillna(0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filled_zero)  # 데이터 표준화
#
# # PCA 모델 설정, n_components는 추출하려는 주성분 수
# pca = PCA()  # 예를 들어, 처음 5개의 주성분만 추출
# pca.fit(scaled_data)
# # principal_components = pca.fit_transform(scaled_data)
# cumulative_variance = pca.explained_variance_ratio_.cumsum()

# plt.figure(figsize=(8, 4))
# plt.plot(cumulative_variance)
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.show()

pca = PCA(n_components=20)  # 예를 들어, 5개의 주성분 사용
pca.fit(scaled_data)
transformed_data = pca.transform(scaled_data)

# 각 센서가 주성분에 기여하는 정도 확인
component_contributions = pca.components_
importance_df = pd.DataFrame(component_contributions, columns=data.columns)

# 가장 기여도가 높은 센서 확인
top_sensors = importance_df.abs().idxmax(axis=1)
print(top_sensors)