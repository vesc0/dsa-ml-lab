import numpy as np
import pandas as pd

def loss(X, y, w, lambda_reg):
	err = X @ w - y
	return (err ** 2).mean() + lambda_reg * (w[1:] ** 2).sum()

def best_alpha(X, y, w, grad, alpha, lambda_reg):
	base = loss(X, y, w, lambda_reg)
	while alpha > 1e-12 and loss(X, y, w - alpha * grad, lambda_reg) > base:
		alpha /= 2
	return alpha

#%% UCENJE
def learn(data, target_att=None, alpha=0.1, lambda_reg=0.0, stochastic=False, auto_alpha=False, max_iter=10000, tolerance=0.01):
	target_att = target_att or data.columns[-1]
	y = data[[target_att]].to_numpy(dtype=float)
	X = data.drop(target_att, axis=1)

	X_mean = X.mean()
	X_std = X.std().replace(0, 1)
	X = (X - X_mean) / X_std
	X.insert(0, 'X0', 1.0)
	X = X.to_numpy(dtype=float)

	m, n = X.shape
	w = np.zeros((n, 1))
	rng = np.random.default_rng(0)

	for _ in range(max_iter):
		if stochastic:
			i = rng.integers(m)
			Xi, yi = X[i:i + 1], y[i:i + 1]
		else:
			Xi, yi = X, y

		err = Xi @ w - yi
		grad = 2 * Xi.T @ err / len(Xi)
		grad[1:] += 2 * lambda_reg * w[1:]

		step = best_alpha(X, y, w, grad, alpha, lambda_reg) if auto_alpha else alpha
		w = w - step * grad
		alpha = step

		full_err = X @ w - y
		full_grad = 2 * X.T @ full_err / m
		full_grad[1:] += 2 * lambda_reg * w[1:]
		if abs(full_grad).sum() < tolerance:
			break

	return {
		'_target_att': target_att,
		'_features': data.drop(target_att, axis=1).columns,
		'_mean': X_mean,
		'_std': X_std,
		'_weights': w,
	}

#%% PREDVIDJANJE
def predict(model, new_data):
	if isinstance(new_data, pd.Series):
		return float(predict(model, new_data.to_frame().T).iloc[0])

	X = new_data.copy()
	if model['_target_att'] in X.columns:
		X = X.drop(model['_target_att'], axis=1)
	X = X[model['_features']]
	X = (X - model['_mean']) / model['_std']
	X.insert(0, 'X0', 1.0)

	return pd.Series((X.to_numpy(dtype=float) @ model['_weights']).ravel(), index=new_data.index, name='prediction')

#%% KORISCENJE
data = pd.read_csv('data/boston.csv')
model = learn(data, alpha=0.1, lambda_reg=0.01, auto_alpha=True)

data_new = data.head().copy()
data_new['prediction'] = predict(model, data_new)

print(data_new[[data.columns[-1], 'prediction']])
