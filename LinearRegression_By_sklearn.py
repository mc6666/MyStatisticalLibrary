import numpy as np
import pandas as pd

# 載入 scikit-learn 內建的房屋估價資料集
from sklearn.datasets import load_boston
boston_dataset = load_boston()
X=boston_dataset.data
y = boston_dataset.target

# 計算 Beta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 訓練模型
lin_model = LinearRegression()
lin_model.fit(X, y)

# 計算RMSE、判定係數 
from sklearn.metrics import r2_score
y_predict = lin_model.predict(X)
rmse = (np.sqrt(mean_squared_error(y, y_predict)))
r2 = r2_score(y, y_predict)

print(rmse)
print(r2)
