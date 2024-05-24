import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

# pandas 출력 설정
pd.set_option('display.max_rows', None)  # 모든 행 출력
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', None)  # 셸 너비에 맞춰 출력
pd.set_option('display.max_colwidth', None)  # 열의 최대 너비 제한 없앰

# 데이터 불러오기
data = pd.read_csv('./data/archive/merged_data2.csv')

# 결측치 대체
data = data.dropna(axis=1, how='all')
valid_means = data.mean(skipna=True)
data.fillna(valid_means, inplace=True)

# 이상치를 중앙값으로 대체
for col in data.columns:
    if np.issubdtype(data[col].dtype, np.number):
        median = data[col].median()
        std_dev = data[col].std()
        outliers = (data[col] - median).abs() > 3 * std_dev
        data[col][outliers] = np.nan
        data[col].fillna(median, inplace=True)

# 학습 및 테스트 데이터 분할
X = data.drop('Sensor-1', axis=1)
y = data['Sensor-1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
#
# # 파이프라인 설정
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svr', SVR())
# ])
#
# C_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 800]
# gamma_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
# epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#
# # 하이퍼파라미터 그리드 설정
# param_grid = {
#     'svr__C': C_values,
#     'svr__gamma': gamma_values,
#     'svr__epsilon': epsilon_values
# }
#
# # 그리드 서치 설정
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
# grid_search.fit(X_train, y_train)
#
# # 최적의 파라미터 출력
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))
#
# # 데이터 스케일링과 SVR 모델을 파이프라인으로 구성
# best_params = grid_search.best_params_
# pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=best_params['svr__C'], gamma=best_params['svr__gamma'], epsilon=best_params['svr__epsilon']))

# 데이터 스케일링과 SVR 모델을 파이프라인으로 구성
pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=400, gamma=0.01, epsilon=0.01))

# 모델 학습
pipeline.fit(X_train, y_train)

# 학습 데이터에 대한 결정계수 계산
svr_score = pipeline.score(X_train, y_train)
print("Training Score: ", svr_score)

# 테스트 데이터에 대한 결정계수 계산
svr_test_score = pipeline.score(X_test, y_test)
print("Test Score: ", svr_test_score)



# 테스트 데이터에 대한 예측 수행
y_pred = pipeline.predict(X_test)

# 성능 지표 계산
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# 성능 지표 시각화
metrics = ['MAE', 'MSE', 'RMSE', 'R² Score']
values = [mae, mse, rmse, r2]

plt.figure(figsize=(10, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
plt.ylabel('Value')
plt.title('Performance Metrics')
plt.show()


# # 실제 값과 예측 값 비교 그래프
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 대각선
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs. Predicted')
# plt.show()
#
# errors = y_test - y_pred
# plt.figure(figsize=(10, 6))
# plt.hist(errors, bins=50)
# plt.xlabel('Prediction Error')
# plt.ylabel('Frequency')
# plt.title('Prediction Error Distribution')
# plt.show()