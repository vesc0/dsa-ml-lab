import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB

#%% Priprema podataka

data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug']*2-1

X = pd.get_dummies(X)   # dummy coding: pretvaranje kategorickih u numericke
n,m = X.shape

#%% Algoritam: Ucenje
alfas = pd.Series(np.array([1/n]*n), index=data.index)   # tezine instanci: za prvi model su sve instance podjednako vazne

ensemble_size = 5
ensemble = []
weights = np.zeros(ensemble_size)   # tezine modela u ansamblu, koje odredjuju jacinu prilikom glasanja

for t in range(ensemble_size):
	alg = GaussianNB()   # "slabi"/jednostavni model

	model = alg.fit(X,y, sample_weight=alfas)
	predictions = model.predict(X)
	error = (predictions!=y).astype(int)      # greska i-tog modela u predvidjanju
	weighted_error = (error*alfas).sum()       # ukupna (otezana) greska sa tezinama instanci
	w = 1/2 * math.log((1-weighted_error)/weighted_error)   # preracunavanje tezina modela

	ensemble.append(model)
	weights[t] = w

	factor = np.exp(-w*predictions*y)
	alfas = alfas * factor   # preracunavanje novih tezina instanci, po formuli
	
	z = alfas.sum()   # norma za normalizaciju
	alfas = alfas/z


#%% Algoritam: Predvidjanje i Evaluacija
predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
predictions['ensemble'] = np.sign(predictions.dot(weights))

print((predictions.add(y, axis=0).abs()/2).mean()) # tacnost pojedinacnih modela i ansambla (komplementarnost)




