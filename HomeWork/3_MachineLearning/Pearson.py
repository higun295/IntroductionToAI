import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./data/Wafer-Dataset/Prediction_Batch_files/Learning1.csv')

# 상관계수 매트릭스 계산
correlation_matrix = data.corr()

# 상관계수 매트릭스 히트맵 시각화
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Sensor Data Correlation Matrix')
plt.show()

# 상관계수가 0.8 이상인 항목 필터링
high_corr = correlation_matrix[correlation_matrix > 0.8]
print(high_corr)