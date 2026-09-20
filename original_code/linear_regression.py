import pandas as pd
import numpy as np

data = pd.read_csv('data/house.csv')
y = data[['Price']]
X = data.drop('Price', axis=1)

X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

X['X0'] = 1

m,n = X.shape
X = X.to_numpy()
y = y.to_numpy()

#%% INIT
w = np.random.random((1,n))
alpha = 0.6

#%% LEARNING ALGORITHM: Gradient Descent
for it in range(10000):
    pred = X.dot(w.T)
    err = pred - y
    grad = err.T.dot(X) / m
    w = w - alpha*grad

    MSE = err.T.dot(err) / m
    grad_norm = abs(grad).sum()
    print(it, grad_norm, MSE)
    if grad_norm < 0.01: break

# model = w

#%% PREDICTION
data_new = pd.read_csv('data/house_new.csv')
data_new = (data_new - X_mean) / X_std
data_new['X0'] = 1

predictions = data_new.dot(w.T)
