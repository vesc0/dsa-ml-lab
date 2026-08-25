import pandas as pd
import numpy as np

#%% UCENJE
def learn(data):
    model = {}
    n,m = data.shape
    
    # Normalizacija
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    
    # Inicijalizacija
    centroids = data.sample(k).reset_index(drop=True)
    assign = np.zeros((n,1))
    old_quality = float('inf')
    
    for iteration in range(50):
    	quality = np.zeros(k)
    	# 1. dodela tacaka klasterima
    	for i in range(n):
    		slucaj = data.iloc[i]
    		dist = ((slucaj-centroids)**2).sum(axis=1)
    		assign[i] = np.argmin(dist)
    	
    	# 2. preracunavanje centroida
    	for c in range(k):
    		subset = data[assign==c]
    		centroids.loc[c] = subset.mean()
    		quality[c] = subset.var().sum() * len(subset)
    	
    	total_quality = quality.sum()
    	print(iteration, total_quality)
    	if old_quality == total_quality: break
    	old_quality = total_quality
        
    model['centroids'], model['mean'], model['std'] = centroids, data_mean, data_std
    return model
	
#%% TRANSFORMACIJA
def transform(model, data):
    data = (data - model['mean']) / model['std']
    
    assign = np.zeros((len(data), 1))
    
    for i in range(len(data)):
        slucaj = data.iloc[i]
        dist = ((slucaj-model['centroids'])**2).sum(axis=1)
        assign[i] = np.argmin(dist)
    
    return assign

#%% KORISCENJE
data = pd.read_csv('data/life.csv').set_index('country')
k = 3

model = learn(data)
print(model['centroids']*model['std']+model['mean'])
assign = transform(model, data)
data[assign==2]