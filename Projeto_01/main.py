import pandas as pd
import numpy as np
from src import funcao as fc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv('dataset/banco.csv')
x,y = fc.fun(data)
data = data.drop(['CHAVE'], axis = 1)

training_set, validation_set = train_test_split(data, test_size = 0.2, random_state = 21)
X_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values

classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_pred, y_val)
print('Accuracy of MLPClassifier : ', fc.accuracy(cm))

fc.plot(data)





