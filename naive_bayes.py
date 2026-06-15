import pandas as pd

#%% UCENJE
def learn(data, class_attr):
    model = {}
    
    apriori = data[class_attr].value_counts()
    apriori /= apriori.sum()
    model['_apriori'] = apriori
    
    for attribute in data.drop(class_attr, axis=1).columns:
        mat_cont = pd.crosstab(data[attribute], data[class_attr])
        mat_cont /= mat_cont.sum()
        model[attribute] = mat_cont
    
    return model

#%% PREDIKCIJA
def predict(model, new_instance):
    class_probabilities = {}
    
    for class_value in model['_apriori'].index:
        probability = 1
        
        for attribute in model:
            if attribute == '_apriori':
                probability *= model['_apriori'][class_value]
            else:
                probability *= model[attribute][class_value][new_instance[attribute]]
        class_probabilities[class_value] = probability
    
    prediction = max(class_probabilities, key=class_probabilities.get)
    
    return prediction, class_probabilities

#%% KORISCENJE
data = pd.read_csv('data/prehlada.csv')
model = learn(data, 'Prehlada')

data_new = pd.read_csv('data/prehlada_novi.csv')
for i in range(len(data_new)):
    prediciton, confidence = predict(model, data_new.iloc[i])
    
    data_new.loc[i, 'prediction'] = prediciton
    for klasa in confidence:
        data_new.loc[i, 'class='+klasa] = confidence[klasa]

print(data_new)