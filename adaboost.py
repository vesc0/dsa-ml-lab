import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split

#%% UCENJE
def learn(X, y, base_alg=GaussianNB(), ensemble_size=5, learning_rate=1.0, rng=None):
    n,m = X.shape
    
    rng = np.random.default_rng(rng)
    algs = base_alg if isinstance(base_alg, list) else [base_alg]
    
    alfas = pd.Series(np.array([1/n]*n), index=X.index)   # tezine instanci: za prvi model su sve instance podjednako vazne # data.index -> X.index
    
    ensemble = []
    weights = np.zeros(ensemble_size)   # tezine modela u ansamblu, koje odredjuju jacinu prilikom glasanja
    
    for t in range(ensemble_size):
    	#alg = GaussianNB()   # "slabi"/jednostavni model
        #alg = clone(base_alg)
        alg = clone(algs[rng.integers(len(algs))])
    
        model = alg.fit(X,y, sample_weight=alfas)
        predictions = model.predict(X)
        error = (predictions!=y).astype(int)      # greska i-tog modela u predvidjanju
        #weighted_error = (error*alfas).sum()       # ukupna (otezana) greska sa tezinama instanci
        weighted_error = np.clip((error*alfas).sum(), 1e-10, 1-1e-10)
        w = 1/2 * math.log((1-weighted_error) / weighted_error)   # preracunavanje tezina modela
    
        ensemble.append(model)
        weights[t] = w
    
        factor = np.exp(-learning_rate * w * predictions * y)
        alfas = alfas * factor   # preracunavanje novih tezina instanci, po formuli
    	
        z = alfas.sum()   # norma za normalizaciju
        alfas = alfas/z
        
    return ensemble, weights

#%% PREDVIDJANJE
def predict(X, ensemble, weights):
    predictions = pd.DataFrame([model.predict(X) for model in ensemble], columns=X.index).T
    H = predictions.dot(weights)
    predictions['ensemble'] = np.sign(H)
    predictions['confidence'] = H.abs() / np.abs(weights).sum()
    
    #print((predictions.add(y, axis=0).abs()/2).mean()) # tacnost pojedinacnih modela i ansambla (komplementarnost)

    return predictions

#%% KORISCENJE
# Priprema podataka
data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug'] * 2 - 1

X = pd.get_dummies(X)   # dummy coding: pretvaranje kategorickih u numericke

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# Algoritam: Ucenje
algs = [GaussianNB(), DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]
ensemble, weights = learn(X_tr, y_tr, algs, ensemble_size=10, learning_rate=1.0, rng=42)

# Algoritam: Predvidjanje i Evaluacija
pr = predict(X_te, ensemble, weights)

print('tacnost:', (pr['ensemble'] == y_te).mean())
t = pr['ensemble'] == y_te
print('conf tacno :', pr.loc[t, 'confidence'].mean())
print('conf greska:', pr.loc[~t, 'confidence'].mean())
