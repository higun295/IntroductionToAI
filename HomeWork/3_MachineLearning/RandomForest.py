import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기 및 전처리
data = pd.read_csv('./data/archive/merged_data2.csv')
data = data.dropna(axis=1, how='all')
valid_means = data.mean(skipna=True)
data.fillna(valid_means, inplace=True)

# 이상치를 중앙값으로 대체
for col in data.columns:
    if np.issubdtype(data[col].dtype, np.number):
        median = data[col].median()
        std_dev = data[col].std()
        outliers = (data[col] - median).abs() > 3 * std_dev
        data.loc[outliers, col] = np.nan
        data[col].fillna(median, inplace=True)

# 학습 및 테스트 데이터 분할
X = data.drop('Sensor-1', axis=1)
y = data['Sensor-1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# 데이터 스케일링과 랜덤 포레스트 모델을 파이프라인으로 구성
pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42, verbose=2))

# 그리드서치 파라미터 설정
param_grid = {
    'randomforestregressor__n_estimators': [100, 200],
    'randomforestregressor__max_depth': [10, 20, 30],
    'randomforestregressor__min_samples_split': [2, 4],
    'randomforestregressor__min_samples_leaf': [1, 2]
}

# 그리드서치 객체 생성
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

# 그리드서치 실행
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최적의 모델 성능 출력
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# 테스트 데이터에 대한 성능 평가
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Score: ", test_score)