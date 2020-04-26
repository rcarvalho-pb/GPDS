import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn
from src import funcao as fc


data = pd.read_csv('dataset/banco.csv')
data.columns = data.columns.str.strip()

x,y = fc.fun(data)

#Splitting the dataset into  training and validation sets
from sklearn.model_selection import train_test_split
training_set, validation_set = train_test_split(data, test_size = 0.2, random_state = 21)

print(training_set)
print(validation_set)

X_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values
print(X_train)
print(y_train)
print(X_val)
print(y_val)

#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)
#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_val)
print('Accuracy of MLPClassifier : ', fc.accuracy(cm))




