import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

# 載入 scikit-learn 內建的房屋估價資料集
boston_dataset = load_boston()
X = boston_dataset.data
# 設定 b 對應的 X，固定為 1
b=np.ones((X.shape[0], 1))

# X 結合 b 對應的 X
X=np.hstack((X, b))

# y 改為二維，以利矩陣運算
y = boston_dataset.target
y = y.reshape((-1, 1))

# 計算 Beta
Beta = np.linalg.inv(X.T @ X) @ X.T @ y

# 計算RMSE， RMSE = MSE 開根號 
SSE = ((X @ Beta - y ) ** 2).sum() 
MSE = SSE / y.shape[0]
RMSE = MSE ** (1/2)
print(RMSE)

# 計算判定係數
y_mean = y.ravel().mean()
SST = ((y - y_mean) ** 2).sum()
R2 = 1 - (SSE / SST)
print(R2)