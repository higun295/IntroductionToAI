# 실제 데이터를 기반으로 머신러닝 알고리즘 적용
# 데이터는 많으면 많을수록 좋음 (100개 이상)

# [miniproject 보고서 포함 내용]
# - 데이터 설명
# - 사용한 모델
# - 학습 과정
# - 성능
# - 프로젝트 수행을 통해서 얻은 경험

# 1. 데이터 선택
# 데이터 유형: 먼저 분석 과제의 주제를 이해하고 적절한 데이터 유형을 선택합니다. 예를 들어, 분류 과제는 레이블이 있는 데이터가 필요하며, 회귀 과제는 연속형 타깃 변수를 가진 데이터를 요구합니다.
# 데이터 출처: Kaggle, UCI Machine Learning Repository, Google Dataset Search 등을 활용하면 공공 데이터셋을 찾을 수 있습니다.
# 데이터 전처리: 데이터에 결측값, 이상값, 중복값 등을 처리해야 합니다. 또한 데이터의 분포를 시각화하고 특징을 파악하는 것도 중요합니다.
    # 1. 결측값 처리
    # 결측값 확인: pandas 라이브러리의 .isnull().sum()을 사용해 각 컬럼의 결측값 수를 파악합니다.
    # 결측값 처리 방법:
        # 삭제: 특정 행이나 열의 결측값이 많을 경우, 해당 행이나 열을 삭제할 수 있습니다.
        # 대체: 평균, 중간값, 최빈값으로 결측값을 대체하거나, 회귀 모델을 사용해 결측값을 예측하는 방법도 있습니다.
    # 2. 이상값 처리
    # 이상값 확인: describe() 함수로 통계량을 확인하거나, 박스 플롯을 사용해 이상값을 시각화합니다.
            # 1. 이상값 확인
            # 이상값을 확인하는 한 가지 방법은 박스플롯을 사용하는 것입니다. matplotlib와 seaborn 라이브러리를 활용하여 각 센서의 데이터 분포를 시각화할 수 있습니다.
    # 이상값 처리 방법:
        # 삭제: 이상값을 가진 행을 삭제하거나, 필터링하여 제한된 범위의 데이터만 사용합니다.
            # 2. 이상값 처리
            # 이상값을 처리하는 방법은 여러 가지가 있습니다. 가장 일반적인 방법 중 하나는 IQR(Interquartile Range)을 사용하는 것입니다. IQR은 제1사분위수(Q1)와 제3사분위수(Q3)의 차이로, 이를 이용해 이상값의 범위를 설정할 수 있습니다.
            # 이상값이 발견되면, 이 값을 제거하거나 다른 값으로 대체할 수 있습니다. 위 예제에서는 이상값을 가진 행 전체를 제거했습니다. 이상값을 NaN으로 설정하거나 다른 방식으로 대체할 수도 있습니다.
            # 이 과정을 통해 데이터의 이상값을 효과적으로 관리하고, 데이터 분석의 정확도를 높일 수 있습니다. 각 데이터셋의 특성에 따라 이상값의 처리 방법과 기준을 조정할 필요가 있습니다.
    # 3. 중복값 처리
    # 중복값 확인 및 제거
    # 4. 데이터 분포 및 특징 시각화
    # 히스토그램: 각 컬럼의 분포를 확인합니다.
    # 상관관계 히트맵: 각 특성 간의 상관관계를 시각화합니다.
# 2. 알고리즘 선택
# 모델 유형: 문제의 성격에 맞는 알고리즘을 선택합니다. 예를 들어, 분류 문제에는 랜덤 포레스트, 로지스틱 회귀, XGBoost 등을 고려할 수 있고, 회귀 문제에는 선형 회귀, 의사결정나무 등을 사용할 수 있습니다.
# 모델 튜닝: 최적의 하이퍼파라미터를 찾기 위해 GridSearchCV, RandomizedSearchCV 등의 방법을 활용해 튜닝합니다.
# 3. 학습 과정
# 데이터 분할: 학습, 검증, 테스트 데이터로 분할합니다. 일반적으로 전체 데이터의 70-80%를 학습 데이터로, 나머지를 검증과 테스트 데이터로 나눕니다.
# 모델 훈련: 학습 데이터로 모델을 훈련시키고, 검증 데이터로 초기에 모델의 성능을 확인합니다.
# 특성 중요도 분석: 일부 알고리즘에서는 피처 중요도를 제공하므로, 이를 통해 어떤 변수가 결과에 큰 영향을 미치는지 파악합니다.
# 4. 성능 평가
# 평가 지표: 문제 유형에 맞는 적절한 평가 지표를 사용합니다.
# 분류 문제: 정확도, 정밀도, 재현율, F1 스코어, ROC-AUC 등을 고려합니다.
# 회귀 문제: 평균 제곱 오차(MSE), 평균 절대 오차(MAE), 결정 계수(R²) 등이 유용합니다.
# 교차 검증: 데이터가 편향되지 않았는지 확인하기 위해 교차 검증을 수행합니다.
# 오버피팅 검사: 훈련 데이터와 검증 데이터에서의 성능 차이를 통해 오버피팅 여부를 확인합니다.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 경로
file_path = './data/Wafer-Dataset/Prediction_Batch_files/Wafer_Test2.csv'

# # pandas를 사용하여 CSV 파일 읽기
data = pd.read_csv(file_path)

# # 각 컬럼별 결측값의 수 확인
# print(data.isnull().sum())

# 결측값을 0으로 대체하기
data_filled_zero = data.fillna(0)
# # 결측값을 평균으로 대체하기
# data_filled_zero = data.fillna(data.mean(), inplace=True)


# # 결측값 처리 후 다시 확인
# print(data_filled.isnull().sum())

# # 박스플롯을 사용하여 이상값 확인
# plt.figure(figsize=(80, 20))
# sns.boxplot(data=data_filled_zero)
# plt.xticks(rotation=90)  # x축 레이블 회전
# plt.show()

# # 히스토그램으로 각 센서의 데이터 분포 확인
# plt.figure(figsize=(20, 15))  # 차트 크기 조정
# data_filled_zero.hist(bins=50)
# plt.show()

# 상관관계 히트맵 생성
plt.figure(figsize=(20, 15))  # 차트 크기 조정
correlation_matrix = data_filled_zero.corr()  # 상관관계 계산
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# # IQR을 계산하고 이상값 필터링
# Q1 = data_filled_zero.quantile(0.25)
# Q3 = data_filled_zero.quantile(0.75)
# IQR = Q3 - Q1
#
# # 이상값의 범위를 정의 (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
# outlier_step = 2.5 * IQR
#
# # 이상값을 NaN으로 설정
# data_filled_zero = data_filled_zero[~((data_filled_zero < (Q1 - outlier_step)) | (data_filled_zero > (Q3 + outlier_step))).any(axis=1)]
#
# # # 이상값을 처리한 후의 데이터 확인
# # print(data_filled_zero.head())
#
# # # 이상값을 처리한 후의 데이터 확인
# plt.figure(figsize=(80, 10))
# sns.boxplot(data=data_filled_zero)
# plt.xticks(rotation=90)  # x축 레이블 회전
# plt.show()

# # 피어슨 상관계수 계산
# correlation_matrix = data.corr()
#
# # 히트맵 시각화
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()