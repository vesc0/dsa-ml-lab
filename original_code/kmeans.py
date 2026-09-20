import pandas as pd
import numpy as np

data = pd.read_csv('data/life.csv').set_index('country')
 

k = 3
n,m = data.shape

# Normalization
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean) / data_std

# Initialization
centroids = data.sample(k).reset_index(drop=True)
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
    print(iteration, total_quality)
    if old_quality == total_quality: break
    old_quality = total_quality

centroids * data_std + data_mean
print(centroids)
data[assign==2]


