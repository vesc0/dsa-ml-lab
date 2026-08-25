import pandas as pd
import numpy as np

#%% UCENJE
def learn(X, y):
    model = {}
    
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std

    X['X0'] = 1

    m,n = X.shape
    X = X.to_numpy()     
    y = y.to_numpy()
    
    # INIT
    w = np.random.random((1,n))
    alpha = 0.6
    
    # ALGORITAM ZA UCENJE: Gradient Descent
    for it in range(10000):
    	pred = X.dot(w.T)
    	err = pred - y
    	grad = err.T.dot(X) / m
    	w = w - alpha*grad
    	
    	MSE = err.T.dot(err) / m
    	grad_norm = abs(grad).sum()
    	print(it, grad_norm, MSE)
    	if grad_norm < 0.01: break
    
    model['w'] = w
    model['mean'] = X_mean
    model['std'] = X_std
    
    return model

#%% PREDVIDJANJE
def predict(model, X):
    X = (X - model['mean']) / model['std']
    X['X0'] = 1
    
    return X.to_numpy().dot(model['w'].T)
    
#%% KORISCENJE
data = pd.read_csv('data/house.csv')
X = data.drop('Price', axis=1)
y = data[['Price']]
model = learn(X, y)

data_new = pd.read_csv('data/house_new.csv')
data_new['price'] = predict(model, data_new)
print(data_new)
