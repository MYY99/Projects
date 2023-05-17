# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:48:49 2022

@author: User
"""

from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import load_iris
import numpy as np
# from sklearn import tree
import joblib
from data_preprocess import clean_data
import copy

dir = 'weather/'
# dataset_van = read_csv(dir+"vancouver.csv", encoding= 'unicode_escape')
# dataset_la = read_csv(dir+"los_angeles.csv", encoding= 'unicode_escape')
dataset_jer = read_csv(dir+"jerusalem.csv", encoding= 'unicode_escape')
dataset_eil = read_csv(dir+"eilat.csv", encoding= 'unicode_escape')
dataset_tad = read_csv(dir+"tel_aviv_district.csv", encoding= 'unicode_escape')
dataset_nah = read_csv(dir+"nahariyya.csv", encoding= 'unicode_escape')
dataset_hai = read_csv(dir+"haifa.csv", encoding= 'unicode_escape')
dataset_bee = read_csv(dir+"beersheba.csv", encoding= 'unicode_escape')
# dataset_sf = read_csv(dir+"san_francisco.csv", encoding= 'unicode_escape')
# dataset_sea = read_csv(dir+"seattle.csv", encoding= 'unicode_escape')
# dataset_sd = read_csv(dir+"san_diego.csv", encoding= 'unicode_escape')
# print(dataset_jer)

#%%
dataset = pd.concat([dataset_jer, dataset_eil, dataset_tad, dataset_nah, dataset_hai, dataset_bee]) #pd.concat([dataset_van, dataset_la, dataset_jer, dataset_eil, dataset_sf, dataset_sea, dataset_sd], axis=0, ignore_index=True)

#%%
dataset = dataset.drop(dataset[dataset.Weather.str.contains('snow')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('volcanic')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('sleet')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('dust')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('fog')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('haze')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('mist')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('smoke')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('squalls')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('thunderstorm')].index)
dataset = dataset.drop(dataset[dataset.Weather.str.contains('sand')].index)
# dataset.loc[dataset.Weather.str.contains('thunderstorm'), 'Weather'] = 'thunderstorm'
dataset.loc[dataset.Weather.str.contains('rain'), 'Weather'] = 'rain'
dataset.loc[dataset.Weather.str.contains('drizzle'), 'Weather'] = 'rain'
dataset.loc[dataset.Weather.str.contains('clouds'), 'Weather'] = 'clouds'

#%%
n = 12000
msk = dataset.groupby('Weather')['Weather'].transform('size') >= n
dataset = pd.concat((dataset[msk].groupby('Weather').sample(n=n), dataset[~msk]), ignore_index=True)

#%%
classes = dataset.loc[:, 'Weather']
data = dataset.loc[:,['Temperature (°C)', 'Humidity']]



#%%
X = data.values
y = classes.values
print(np.unique(y))
print(len(y))

#%%
# for i in range(1000):
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, shuffle=True, random_state = 0)
model_CARTS = DecisionTreeClassifier(criterion="entropy")
model_CARTS.fit(X_train, y_train)
predictions_CARTS = model_CARTS.predict(X_validation)
predictions_CARTS = model_CARTS.predict(X_train)
print(accuracy_score(y_train, predictions_CARTS))
predictions_CARTS = model_CARTS.predict(X_validation) 
print(accuracy_score(y_validation, predictions_CARTS))
print(confusion_matrix(y_validation, predictions_CARTS))
print(classification_report(y_validation, predictions_CARTS))
# break
    
#%%
joblib.dump(model_CARTS, 'model_CARTS_weather.sav')

#%%
model_CARTS.predict(X_validation[0,:].reshape(1,-1))

# #%%
# model_MLP = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='tanh', max_iter = 10000, learning_rate = 'adaptive', learning_rate_init = 0.0001)
# model_MLP.fit(X_train, y_train)
# train_acc = model_MLP.score(X_train, y_train)
# val_acc = model_MLP.score(X_validation, y_validation)
# print('val acc train acc: ', val_acc, train_acc)

# #%%
# cm = confusion_matrix(y_validation, model_MLP.predict(X_validation), labels=np.unique(y))
# print(cm)
# #%%
# np.savetxt('confusion_matrix.txt', cm, fmt='% 4d')

#%%
