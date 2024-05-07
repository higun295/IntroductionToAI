# 실제 데이터를 기반으로 머신러닝 알고리즘 적용
# 데이터는 많으면 많을수록 좋음 (100개 이상)

# [miniproject 보고서 포함 내용]
# - 데이터 설명
# - 사용한 모델
# - 학습 과정
# - 성능
# - 프로젝트 수행을 통해서 얻은 경험

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 경로
file_path = './data/Train2.csv'

# pandas를 사용하여 CSV 파일 읽기
data = pd.read_csv(file_path)

# 피어슨 상관계수 계산
correlation_matrix = data.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()