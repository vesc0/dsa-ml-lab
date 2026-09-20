import pandas as pd
import numpy as np

#%% LEARNING
def learn(data, w, k, N=1):
    if N > 1:
        return min([learn(data, w, k, 1) for _ in range(N)], key=lambda m: np.nan_to_num(m['quality'], nan=np.inf))
    
    model = {}
    n,m = data.shape
    
    # Normalization
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    
    data = data * np.sqrt(w)
    
    # Initialization
    idx = [np.random.randint(n)]
    dmin = ((data - data.iloc[idx[0]])**2).sum(axis=1)
    for _ in range(k-1):
        idx.append(dmin.argmax())
        dmin = np.minimum(dmin, ((data - data.iloc[idx[-1]])**2).sum(axis=1))
    centroids = data.iloc[idx].reset_index(drop=True)

    assign = np.zeros((n,1))
    old_quality = float('inf')
    
    for iteration in range(50):
        quality = np.zeros(k)
        # 1. assign points to clusters
        for i in range(n):
            instance = data.iloc[i]
            dist = ((instance-centroids)**2).sum(axis=1)
            assign[i] = np.argmin(dist)
        
        # 2. recompute centroids
        for c in range(k):
            subset = data[assign==c]
            centroids.loc[c] = subset.mean()
            quality[c] = subset.var().sum() * len(subset)
        
        total_quality = quality.sum()
        #print(iteration, total_quality)
        if old_quality == total_quality: break
        old_quality = total_quality
        
    model['centroids'], model['mean'], model['std'], model['w'], model['quality'] = centroids, data_mean, data_std, w, total_quality
    return model

#%% TRANSFORMATION
def transform(model, data, max_radius=0.8, min_separation=1.0):
    data = (data - model['mean']) / model['std'] * np.sqrt(model['w'])
    c, scale = model['centroids'], np.sqrt(model['w'].sum())
    assign = np.zeros((len(data), 1))
    
    for i in range(len(data)):
        instance = data.iloc[i]
        dist = ((instance-model['centroids'])**2).sum(axis=1)
        assign[i] = np.argmin(dist)
        
    # Diagnostics
    radius = np.array([np.sqrt(((data[assign==i] - c.loc[i])**2).sum(axis=1).mean()) for i in range(len(c))])
    for i in range(len(c)):
        if radius[i] > max_radius * scale:
            print(f'! Cluster {i} is poorly represented by its centroid (r={radius[i]:.2f}, threshold={max_radius*scale:.2f})')
        for j in range(i+1, len(c)):
            d = np.sqrt(((c.loc[i] - c.loc[j])**2).sum())
            if d < min_separation * (radius[i] + radius[j]):
                print(f'! Clusters {i} and {j} are too similar (d={d:.2f}, sum of radii={radius[i]+radius[j]:.2f})')
                
    return assign

#%% USAGE
data = pd.read_csv('data/boston.csv')

# Preparation: numeric columns only, no missing values or constants
data = data.select_dtypes(include=np.number).dropna()
data = data.loc[:, data.std() > 0]

# By default all columns are equally important
w = pd.Series(1.0, index=data.columns)
# w['LSTAT'] = 3
# w['CHAS'] = 0

model = learn(data, w, k=3, N=10)
print(model['centroids'] / np.sqrt(w.replace(0,1)) * model['std'] + model['mean'])

assign = transform(model, data)
print(pd.Series(assign.flatten()).value_counts())
