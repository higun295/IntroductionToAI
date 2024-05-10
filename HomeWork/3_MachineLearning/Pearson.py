import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# PCA 변환 데이터 불러오기
data = pd.read_csv('./data/transformed_data.csv')

# 독립 변수와 종속 변수 분리 (여기서는 예를 들어 'target'이라는 컬럼이 타깃 변수라고 가정)
X = data.drop('target', axis=1)
y = data['target']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model training
model.fit(X_train, y_train)

# 훈련된 모델 저장
joblib.dump(model, 'rf_model.pkl')

# Predicting on test data
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')
#
# # 성능 평가
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
#
# # 특성 중요도 추출 및 시각화
# feature_importances = model.feature_importances_
# plt.barh(range(len(feature_importances)), feature_importances)
# plt.yticks(range(len(X.columns)), X.columns)
# plt.xlabel('Feature Importance')
# plt.show()