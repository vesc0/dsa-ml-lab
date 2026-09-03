import pandas as pd
import numpy as np

#%% UCENJE
def learn(X, y, lam=0, stochastic=False, auto_alpha=False, alpha=None):
    model = {}
    X = X.copy()
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std

    X['X0'] = 1

    m,n = X.shape
    X = X.to_numpy()     
    y = y.to_numpy()
    
    reg_mask = np.ones((1, n))
    reg_mask[0, -1] = 0
    
    # INIT
    w = np.random.random((1,n))
    if alpha is None:
        alpha = 0.05
    prev_cost = np.inf
    w_prev = w.copy()
    alpha0 = alpha
    
    # ALGORITAM ZA UCENJE: Gradient Descent
    for it in range(10000):
        if stochastic:
            j = np.random.randint(m)
            Xb = X[j:j+1]      # oblik (1, n)
            yb = y[j:j+1]      # oblik (1, 1)
        else:
            Xb = X
            yb = y
        pred = Xb.dot(w.T)
        err = pred - yb
        grad = err.T.dot(Xb) / len(Xb) #m
        grad = grad + 2 * lam * w * reg_mask
        
        MSE = err.T.dot(err) / len(Xb) #m
        penalty = lam * ((w * reg_mask) ** 2).sum()
        cost = MSE + penalty
        
        if auto_alpha and not stochastic:
            c = cost.item()
            if c > prev_cost:
                w = w_prev # ponisti prosli korak
                prev_cost = np.inf # sledeci korak neka prodje
                alpha *= 0.5
                continue
            else:
                alpha *= 1.05
                prev_cost = c
                w_prev = w.copy()
        
        if auto_alpha and stochastic:
            alpha = alpha0 / (1 + it * 0.001)
        
        w = w - alpha*grad
        
        grad_norm = abs(grad).sum()
        print(it, grad_norm, MSE, cost)
        if not stochastic and grad_norm < 0.01: break
    
    model['w'] = w
    model['mean'] = X_mean
    model['std'] = X_std
    
    return model

#%% PREDVIDJANJE
def predict(model, X):
    X = X.copy()
    X = (X - model['mean']) / model['std']
    X['X0'] = 1
    
    return X.to_numpy().dot(model['w'].T)
    
#%% KORISCENJE
data = pd.read_csv('data/boston.csv')

# Podela skupa na trening i test
data = data.sample(frac=1, random_state=42)
split = int(0.8 * len(data))
train = data[:split]
test  = data[split:]

X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]
X_test  = test.iloc[:, :-1]
y_test  = test.iloc[:, -1:]

# Ucenje na trening skupu
model = learn(X_train, y_train, lam=0.0001, auto_alpha=True)

# Predikcija na testu
pred = predict(model, X_test)
print(pred[:10])
print(y_test.to_numpy()[:10])
