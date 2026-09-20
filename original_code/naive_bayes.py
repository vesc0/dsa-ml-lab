import pandas as pd

#%% LEARNING
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

#%% PREDICTION
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

#%% USAGE
data = pd.read_csv('data/cold.csv')
model = learn(data, 'Cold')

data_new = pd.read_csv('data/cold_new.csv')
for i in range(len(data_new)):
    prediciton, confidence = predict(model, data_new.iloc[i])
    
    data_new.loc[i, 'prediction'] = prediciton
    for class_value in confidence:
        data_new.loc[i, 'class='+class_value] = confidence[class_value]

print(data_new)