import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB

#%% Data preparation

data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug']*2-1

X = pd.get_dummies(X)   # dummy coding: converting categorical to numeric
n,m = X.shape

#%% Algorithm: Learning
alphas = pd.Series(np.array([1/n]*n), index=data.index)   # instance weights: for the first model all instances are equally important

ensemble_size = 5
ensemble = []
weights = np.zeros(ensemble_size)   # model weights in the ensemble, which determine voting strength

for t in range(ensemble_size):
    alg = GaussianNB()   # "weak"/simple model

    model = alg.fit(X,y, sample_weight=alphas)
    predictions = model.predict(X)
    error = (predictions!=y).astype(int)      # error of the i-th model in prediction
    weighted_error = (error*alphas).sum()       # total (weighted) error with instance weights
    w = 1/2 * math.log((1-weighted_error)/weighted_error)   # recomputing the model weights

    ensemble.append(model)
    weights[t] = w

    factor = np.exp(-w*predictions*y)
    alphas = alphas * factor   # recomputing the new instance weights, by the formula

    z = alphas.sum()   # normalization constant
    alphas = alphas/z


#%% Algorithm: Prediction and Evaluation
predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
predictions['ensemble'] = np.sign(predictions.dot(weights))

print((predictions.add(y, axis=0).abs()/2).mean()) # accuracy of the individual models and of the ensemble (complementarity)




