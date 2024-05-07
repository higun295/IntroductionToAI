import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 경로
file_path = '/content/drive/My Drive/Colab Notebooks/Test.csv'

# pandas를 사용하여 CSV 파일 읽기
data = pd.read_csv(file_path)

# 피어슨 상관계수 계산
correlation_matrix = data.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()