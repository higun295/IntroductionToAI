from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./data/Wafer-Dataset/Prediction_Batch_files/Learning1.csv')

# 결측값을 0으로 대체하기
data_filled_zero = data.fillna(0)

# 상관관계 매트릭스 계산
corr_matrix = data_filled_zero.corr()

# 상관관계 절대값을 기준으로 내림차순 정렬
sorted_corr = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)

# 동일한 특성 쌍 제거 및 자기 자신과의 상관관계 제거
sorted_corr = sorted_corr[sorted_corr < 1].drop_duplicates()

# 상위 50개의 특성 선택
top_features = sorted_corr.head(50).index.tolist()

# 해당 특성만 포함하는 데이터 프레임 생성
selected_data = data_filled_zero[[item for sublist in top_features for item in sublist]].drop_duplicates()

# 데이터 표준화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# PCA 모델 생성 및 학습
pca = PCA(n_components=0.95)  # 95%의 분산 설명
principal_components = pca.fit_transform(scaled_data)

# 결과 출력
print("선택된 주성분 수:", pca.n_components_)

# 각 주성분에 대한 기여도 출력
components_df = pd.DataFrame(pca.components_, columns=selected_data.columns)
print(components_df)

# 변환된 주성분 데이터를 DataFrame으로 변환
transformed_df = pd.DataFrame(principal_components)

# 변환된 데이터를 파일로 저장하거나 다른 분석에 사용
transformed_df.to_csv('./data/transformed_data.csv', index=False)

# # 첫 번째 주성분의 센서 기여도 시각화
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(components_df.columns)), components_df.iloc[1])
# plt.xlabel('Sensors')
# plt.ylabel('Contribution to the 1st Principal Component')
# plt.title('Sensor Contributions to the First Principal Component')
# plt.xticks(range(len(components_df.columns)), components_df.columns, rotation=90)
# plt.show()






# # 결측값을 0으로 대체하기
# data_filled_zero = data.fillna(0)
#
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data_filled_zero)  # 데이터 표준화
# #
# # # PCA 모델 설정, n_components는 추출하려는 주성분 수
# # pca = PCA()  # 예를 들어, 처음 5개의 주성분만 추출
# # pca.fit(scaled_data)
# # # principal_components = pca.fit_transform(scaled_data)
# # cumulative_variance = pca.explained_variance_ratio_.cumsum()
#
# # plt.figure(figsize=(8, 4))
# # plt.plot(cumulative_variance)
# # plt.xlabel('Number of Components')
# # plt.ylabel('Cumulative Explained Variance')
# # plt.show()
#
# pca = PCA(n_components=20)  # 예를 들어, 5개의 주성분 사용
# pca.fit(scaled_data)
# transformed_data = pca.transform(scaled_data)
#
# # 각 센서가 주성분에 기여하는 정도 확인
# component_contributions = pca.components_
# importance_df = pd.DataFrame(component_contributions, columns=data.columns)
#
# # 가장 기여도가 높은 센서 확인
# top_sensors = importance_df.abs().idxmax(axis=1)
# print(top_sensors)