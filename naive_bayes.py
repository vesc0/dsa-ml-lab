import math
import pandas as pd
from pandas.api.types import is_numeric_dtype

MIN_STD = 1e-9

def _safe_log(value):
	return math.log(value) if value > 0 else float('-inf')

def _gaussian_log_pdf(x, mean, std):
	variance = std ** 2
	return -0.5 * math.log(2 * math.pi * variance) - ((x - mean) ** 2) / (2 * variance)

def _normalize_log_scores(log_scores):
	max_log = max(log_scores.values())
	exp_scores = {cls: math.exp(score - max_log) for cls, score in log_scores.items()}
	total = sum(exp_scores.values())
	return {cls: score / total for cls, score in exp_scores.items()}

#%% UCENJE
def learn(data, class_att=None, alpha=1.0):
	class_att = class_att or data.columns[-1]
	model = {'_class_att': class_att, '_alpha': alpha, '_attributes': {}}

	class_counts = data[class_att].value_counts().sort_index()
	class_total = class_counts.sum()
	num_classes = len(class_counts)
	model['_class_counts'] = class_counts
	model['_log_apriori'] = ((class_counts + alpha) / (class_total + alpha * num_classes)).apply(math.log)

	for attribute in data.drop(columns=class_att).columns:
		if is_numeric_dtype(data[attribute]):
			stats = data.groupby(class_att)[attribute].agg(['mean', 'std']).reindex(class_counts.index)
			stats['std'] = stats['std'].fillna(0.0).clip(lower=MIN_STD)
			model['_attributes'][attribute] = {'type': 'numeric', 'stats': stats}
			continue

		value_counts = data[attribute].nunique(dropna=True)
		contingency = pd.crosstab(data[attribute], data[class_att]).reindex(columns=class_counts.index, fill_value=0)
		denominator = class_counts + alpha * value_counts
		probabilities = contingency.add(alpha).divide(denominator, axis=1)
		default_prob = alpha / denominator
		model['_attributes'][attribute] = {
			'type': 'categorical',
			'probabilities': probabilities,
			'default_prob': default_prob,
		}

	return model

#%% PREDVIDJANJE
def predict(model, new_instance):
	log_scores = {}

	for class_value in model['_class_counts'].index:
		log_probability = model['_log_apriori'][class_value]

		for attribute, info in model['_attributes'].items():
			value = new_instance[attribute]
			if pd.isna(value):
				continue

			if info['type'] == 'numeric':
				stats = info['stats'].loc[class_value]
				log_probability += _gaussian_log_pdf(float(value), stats['mean'], stats['std'])
			else:
				if value in info['probabilities'].index:
					probability = info['probabilities'].loc[value, class_value]
				else:
					probability = info['default_prob'][class_value]
				log_probability += _safe_log(probability)

		log_scores[class_value] = log_probability

	class_probabilities = _normalize_log_scores(log_scores)
	prediction = max(class_probabilities, key=class_probabilities.get)
	return prediction, class_probabilities


#%% KORISCENJE
data = pd.read_csv('data/drug.csv')
class_att = data.columns[-1]
model = learn(data, class_att=class_att, alpha=1.0)

data_new = data.drop(columns=class_att).copy()

for i in range(len(data_new)):
	prediction, confidence = predict(model, data_new.iloc[i])
	data_new.loc[i, 'prediction'] = prediction
	data_new.loc[i, 'actual_class'] = data.loc[i, class_att]
	for class_name, probability in confidence.items():
		data_new.loc[i, f'class={class_name}'] = probability

print(data_new)