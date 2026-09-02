import pandas as pd
import numpy as np
from scipy.stats import norm

#%% UCENJE
def learn(data, class_attr, smoothing=1):
    model = {}
    model['_types'] = {}
    
    apriori = data[class_attr].value_counts() + smoothing
    apriori /= apriori.sum()
    model['_apriori'] = apriori
    
    for attribute in data.drop(class_attr, axis=1).columns:
        if pd.api.types.is_numeric_dtype(data[attribute]):
            mean_var = data.groupby(class_attr)[attribute].agg(['mean', 'var'])
            model[attribute] = mean_var
            model['_types'][attribute] = 'numeric'
        else:
            mat_cont = pd.crosstab(data[attribute], data[class_attr]) + smoothing
            mat_cont /= mat_cont.sum()
            model[attribute] = mat_cont
            model['_types'][attribute] = 'categorical'
    
    return model

#%% PREDIKCIJA
def predict(model, new_instance):
    class_probabilities = {}
    
    epsilon = 1e-9
    for class_value in model['_apriori'].index:
        log_probability = 0
        
        for attribute in model:
            if attribute == '_types':
                continue
            if attribute == '_apriori':
                p = model['_apriori'][class_value]
            else:
                if model['_types'][attribute] == 'numeric':
                    mean = model[attribute].loc[class_value, 'mean']
                    var = model[attribute].loc[class_value, 'var']
                    var = max(var, epsilon)
                    p = norm.pdf(new_instance[attribute], loc=mean, scale=np.sqrt(var))
                else:
                    # kategoricki
                    p = model[attribute][class_value][new_instance[attribute]]
                
            log_probability += np.log(p) if p > 0 else np.log(epsilon)
            
        class_probabilities[class_value] = log_probability
    
    prediction = max(class_probabilities, key=class_probabilities.get)
    
    return prediction, class_probabilities

#%% KORISCENJE
data = pd.read_csv('data/drug.csv')
class_attr = data.columns[-1]

# Podela skupa na trening i test
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(data) * 0.8)
train, test = data[:split], data[split:].reset_index(drop=True)

# Ucenje na trening skupu
model = learn(train, class_attr, smoothing=0)

# Predikcija na testu
data_new = test
for i in range(len(data_new)):
    prediciton, confidence = predict(model, data_new.iloc[i])
    
    data_new.loc[i, 'prediction'] = prediciton
    for klasa in confidence:
        data_new.loc[i, 'class='+klasa] = confidence[klasa]
