import pandas as pd
#import numpy as np
# pd.set_option('display.width',1000)


#%%  UCENJE
def learn(data, class_att):
	model = {}

	apriori = data[class_att].value_counts()
	apriori = apriori / apriori.sum()
	model['_apriori'] = apriori

	for attribute in data.drop(class_att, axis=1).columns:
		mat_cont = pd.crosstab(data[attribute],data[class_att])
		mat_cont = mat_cont / mat_cont.sum(axis=0)
		model[attribute] = mat_cont

	return model


#%% PREDVIDJANJE
def predict(model, new_instance):
	class_probabilities = {}
	for class_value in model['_apriori'].index:
		probability = 1
		
		for attribute in model:
			if attribute == '_apriori':
				probability = probability * model['_apriori'][class_value]
			else:
				probability = probability * model[attribute][class_value][new_instance[attribute]]
		class_probabilities[class_value] = probability

	prediction = max(class_probabilities, key=class_probabilities.get)

	return prediction, class_probabilities

#%% KORISCENJE
data = pd.read_csv('data/prehlada.csv')
model = learn(data,'Prehlada')

data_new = pd.read_csv('data/prehlada_novi.csv')

for i in range(len(data_new)):
	prediction, confidence = predict(model, data_new.iloc[i])

	data_new.loc[i,'prediction'] = prediction
	for klasa in confidence:
		data_new.loc[i,'class='+klasa] = confidence[klasa]

print(data_new)
