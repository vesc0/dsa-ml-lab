import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split

#%% LEARNING
def learn(X, y, base_alg=GaussianNB(), ensemble_size=5, learning_rate=1.0, rng=None):
    n,m = X.shape
    
    rng = np.random.default_rng(rng)
    algs = base_alg if isinstance(base_alg, list) else [base_alg]
    
    alphas = pd.Series(np.array([1/n]*n), index=X.index) # instance weights: for the first model all instances are equally important
    
    ensemble = []
    weights = np.zeros(ensemble_size) # model weights in the ensemble, which determine voting strength
    
    for t in range(ensemble_size):
        alg = clone(algs[rng.integers(len(algs))])
    
        model = alg.fit(X,y, sample_weight=alphas)
        predictions = model.predict(X)
        error = (predictions!=y).astype(int) # error of the i-th model in prediction
        weighted_error = np.clip((error*alphas).sum(), 1e-10, 1-1e-10) # total (weighted) error with instance weights
        w = 1/2 * math.log((1-weighted_error) / weighted_error) # recomputing the model weights
    
        ensemble.append(model)
        weights[t] = w
    
        factor = np.exp(-learning_rate * w * predictions * y)
        alphas = alphas * factor # recomputing the new instance weights

        z = alphas.sum() # normalization constant
        alphas = alphas/z
        
    return ensemble, weights

#%% PREDICTION
def predict(X, ensemble, weights):
    predictions = pd.DataFrame([model.predict(X) for model in ensemble], columns=X.index).T
    H = predictions.dot(weights)
    predictions['ensemble'] = np.sign(H)
    predictions['confidence'] = H.abs() / np.abs(weights).sum()

    return predictions

#%% USAGE
# Data preparation
data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug'] * 2 - 1
X = pd.get_dummies(X)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# Algorithm: Learning
algs = [GaussianNB(), DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]
ensemble, weights = learn(X_tr, y_tr, algs, ensemble_size=10, learning_rate=1.0, rng=42)

# Algorithm: Prediction and Evaluation
pr = predict(X_te, ensemble, weights)

print('accuracy:', (pr['ensemble'] == y_te).mean())
t = pr['ensemble'] == y_te
print('conf correct:', pr.loc[t, 'confidence'].mean())
print('conf wrong  :', pr.loc[~t, 'confidence'].mean())
