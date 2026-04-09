import math
import pandas as pd
from pandas.api.types import is_numeric_dtype

MIN_STD = 1e-9

#%%  UCENJE
def learn(data, class_att=None, alpha=1.0):
	class_att = class_att or data.columns[-1]
	class_counts = data[class_att].value_counts().sort_index()
	classes = class_counts.index
	model = {
		'_class_att': class_att,
		'_classes': classes,
		'_apriori': ((class_counts + alpha) / (len(data) + alpha * len(classes))).apply(math.log),
		'_attributes': {},
	}

	for attribute in data.drop(class_att, axis=1).columns:
		if is_numeric_dtype(data[attribute]):
			stats = data.groupby(class_att)[attribute].agg(['mean', 'std']).reindex(classes)
			stats['std'] = stats['std'].fillna(0).clip(lower=MIN_STD)
			model['_attributes'][attribute] = {'type': 'num', 'stats': stats}
		else:
			mat = pd.crosstab(data[attribute], data[class_att]).reindex(columns=classes, fill_value=0)
			k = data[attribute].nunique(dropna=True)
			den = class_counts + alpha * k
			model['_attributes'][attribute] = {
				'type': 'cat',
				'p': mat.add(alpha).divide(den, axis=1),
				'default': alpha / den,
			}

	return model

#%% PREDVIDJANJE
def predict(model, new_instance):
	log_scores = {}

	for class_value in model['_classes']:
		score = model['_apriori'][class_value]

		for attribute, info in model['_attributes'].items():
			value = new_instance[attribute]
			if pd.isna(value):
				continue

			if info['type'] == 'num':
				mean, std = info['stats'].loc[class_value]
				score += -0.5 * math.log(2 * math.pi * std ** 2) - (float(value) - mean) ** 2 / (2 * std ** 2)
			else:
				p = info['p'].loc[value, class_value] if value in info['p'].index else info['default'][class_value]
				score += math.log(p) if p > 0 else float('-inf')

		log_scores[class_value] = score

	max_log = max(log_scores.values())
	class_probabilities = {c: math.exp(s - max_log) for c, s in log_scores.items()}
	total = sum(class_probabilities.values())
	class_probabilities = {c: p / total for c, p in class_probabilities.items()}

	return max(class_probabilities, key=class_probabilities.get), class_probabilities

#%% KORISCENJE
data = pd.read_csv('data/drug.csv')
class_att = data.columns[-1]
model = learn(data, class_att, alpha=1.0)

data_new = data.drop(class_att, axis=1).copy()

for i in range(len(data_new)):
	prediction, confidence = predict(model, data_new.iloc[i])
	data_new.loc[i, 'prediction'] = prediction
	data_new.loc[i, 'actual_class'] = data.loc[i, class_att]
	for klasa in confidence:
		data_new.loc[i, 'class=' + str(klasa)] = confidence[klasa]

print(data_new)
