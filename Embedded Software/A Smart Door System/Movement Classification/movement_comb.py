# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:22:15 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:15:17 2022

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
# from sklearn.metrics import classification_reportz
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import load_iris
import numpy as np
# from sklearn import tree
import joblib
from data_preprocess import clean_data
import copy

# %% data load
dir = 'data/' #ROOT + '/data/'
dataset_ba = read_csv(dir+"comb_backward.csv", header=None)
dataset_co = read_csv(dir+"comb_come.csv", header=None)
dataset_fs = read_csv(dir+"comb_far_stat.csv", header=None)
dataset_for = read_csv(dir+"comb_forward.csv", header=None)
dataset_lea = read_csv(dir+"comb_leave.csv", header=None)
dataset_nea = read_csv(dir+"comb_near_stat.csv", header=None)
dataset_no = read_csv(dir+"comb_no_stat.csv", header=None)
dataset_pb = read_csv(dir+"comb_pass_by.csv", header=None)

#%% binary classification (static vs dynamic)
#%%%
dataset = pd.concat([dataset_ba, dataset_co, dataset_fs, dataset_for, dataset_lea, dataset_nea, dataset_no, dataset_pb], axis=0, ignore_index=True)
classes = ['backward', 'come from the side', 'far stat', 'forward', 'leave to the side', 'near stat', 'no stat', 'pass by']
class_data_len = np.array([dataset_ba.shape[0], dataset_co.shape[0], dataset_fs.shape[0], dataset_for.shape[0], dataset_lea.shape[0], dataset_nea.shape[0], dataset_no.shape[0], dataset_pb.shape[0]])

X = dataset.values #np.transpose(dataset.values)
y = np.repeat(classes, class_data_len)

#%%% 
y_2class = copy.deepcopy(y)
y_2class[np.core.defchararray.find(y_2class, 'stat') != -1] = 'static'
y_2class[np.core.defchararray.find(y_2class, 'static') == -1] = 'dynamic' #if np.core.defchararray.find(y, 'stat') != -1 else 'dynamic'

#%%
X, _ = clean_data(X, save_csv=True)

#%%
X_train2, X_validation2, Y_train2, Y_validation2 = train_test_split(X, y_2class, test_size=0.1, shuffle=True, random_state = 0)

#%%
model_MLP2 = MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200, 200), max_iter = 10000, learning_rate = 'adaptive', learning_rate_init = 0.0001)
model_MLP2.fit(X_train2, Y_train2)
train_acc2 = model_MLP2.score(X_train2, Y_train2)
val_acc2 = model_MLP2.score(X_validation2, Y_validation2)
print('val acc train acc: ', val_acc2, train_acc2)
#%%
joblib.dump(model_MLP2, 'model_MLP2.sav')

#%% dynamic 
#%%%
dataset_dyn = pd.concat([dataset_ba, dataset_lea, dataset_co, dataset_for, dataset_pb], axis=0, ignore_index=True)
classes_dyn = ['backward/leave to the side', 'come from the side/forward', 'pass by']
class_data_len_dyn = np.array([dataset_ba.shape[0] + dataset_lea.shape[0], dataset_co.shape[0] + dataset_for.shape[0], dataset_pb.shape[0]])

X_dyn = dataset_dyn.values #np.transpose(dataset.values)
y_dyn = np.repeat(classes_dyn, class_data_len_dyn)

X_train_dyn, X_validation_dyn, Y_train_dyn, Y_validation_dyn = train_test_split(X_dyn, y_dyn, test_size=0.10, shuffle=True, random_state = 0)

#%%%
model_MLP_dyn = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100), alpha = 0.003, max_iter = 10000, learning_rate = 'adaptive', learning_rate_init = 0.0001)
model_MLP_dyn.fit(X_train_dyn, Y_train_dyn)
train_acc_dyn = model_MLP_dyn.score(X_train_dyn, Y_train_dyn)
val_acc_dyn = model_MLP_dyn.score(X_validation_dyn, Y_validation_dyn)
print('val acc train acc: ', val_acc_dyn, train_acc_dyn)

#%%%
joblib.dump(model_MLP_dyn, 'model_MLP_dyn.sav')

#%% static
dataset_sta = pd.concat([dataset_fs, dataset_nea, dataset_no], axis=0, ignore_index=True)
classes_sta = ['far stationary', 'near stationary', 'no stationary']
class_data_len_sta = np.array([dataset_fs.shape[0], dataset_nea.shape[0], dataset_no.shape[0]])
X_sta = dataset_sta.values #np.transpose(dataset.values)
y_sta = np.repeat(classes_sta, class_data_len_sta)
X_sta_avg = np.mean(X_sta, axis=1)
y_pred = np.empty(X_sta_avg.shape, dtype=object)
y_pred[X_sta_avg<100] = 'far stationary'
y_pred[X_sta_avg<50] = 'near stationary'
y_pred[X_sta_avg>=100] = 'no stationary'

#%%
num_true = np.sum(np.equal(y_pred, y_sta))
acc = num_true/len(y_sta)
print(acc)
 
#%%
# #%%
# best_val_acc = 0
# for k in range (6, 50, 2):
#     for j in range (6, 50, 2):
#         for i in range (6, 50, 2):
#             model_MLP = MLPClassifier(hidden_layer_sizes=(i+1, j+1, k+1), max_iter = 10000, learning_rate = 'adaptive', learning_rate_init = 0.05)
#             model_MLP.fit(X_train, Y_train)
#             train_acc = model_MLP.score(X_train, Y_train)
#             val_acc = model_MLP.score(X_validation, Y_validation)
#             print('val acc train acc: ', i, j, k, val_acc, train_acc)
#             if best_val_acc < val_acc:
#                 best_val_acc = val_acc 
#                 print('Best validation accuracy training accuracy: ', i+1, j+1, k+1, best_val_acc, train_acc)

# #%%
# print("KNN3" + spec)
# # KNN3
# model_KNN = KNeighborsClassifier(n_neighbors=3)
# model_KNN.fit(X_train, Y_train)
# predictions_KNN = model_KNN.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_KNN))
# print(confusion_matrix(Y_validation, predictions_KNN))
# print(classification_report(Y_validation, predictions_KNN))

# print("KNN5" + spec)
# # KNN5
# model_KNN = KNeighborsClassifier(n_neighbors=5)
# model_KNN.fit(X_train, Y_train)
# predictions_KNN = model_KNN.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_KNN))
# print(confusion_matrix(Y_validation, predictions_KNN))
# print(classification_report(Y_validation, predictions_KNN))

# print("CARTS (gini)" + spec)
# # CARTS gini
# model_CARTS = DecisionTreeClassifier(criterion="gini")
# model_CARTS.fit(X_train, Y_train)
# predictions_CARTS = model_CARTS.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_CARTS))
# print(confusion_matrix(Y_validation, predictions_CARTS))
# print(classification_report(Y_validation, predictions_CARTS))

# print("CARTS (entropy)" + spec)
# # CARTS entropy
# model_CARTS = DecisionTreeClassifier(criterion="entropy")
# model_CARTS.fit(X_train, Y_train)
# predictions_CARTS = model_CARTS.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_CARTS))
# print(confusion_matrix(Y_validation, predictions_CARTS))
# print(classification_report(Y_validation, predictions_CARTS))

# print("SVM (rbf)" + spec)
# # SVM rbf
# model_SVM = SVC(gamma='auto', kernel='rbf')
# model_SVM.fit(X_train, Y_train)
# predictions_SVM = model_SVM.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_SVM))
# print(confusion_matrix(Y_validation, predictions_SVM))
# print(classification_report(Y_validation, predictions_SVM))

# print("SVM (linear)" + spec)
# # SVM linear
# model_SVM = SVC(gamma='auto', kernel='linear')
# model_SVM.fit(X_train, Y_train)
# predictions_SVM = model_SVM.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_SVM))
# print(confusion_matrix(Y_validation, predictions_SVM))
# print(classification_report(Y_validation, predictions_SVM))

# print("SVM (sigmoid)" + spec)
# # SVM sigmoid
# model_SVM = SVC(gamma='auto', kernel='sigmoid')
# model_SVM.fit(X_train, Y_train)
# predictions_SVM = model_SVM.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_SVM))
# print(confusion_matrix(Y_validation, predictions_SVM))
# print(classification_report(Y_validation, predictions_SVM))

# print("SVM (poly2)" + spec)
# # SVM poly2
# model_SVM = SVC(gamma='auto', kernel='poly', degree=2)
# model_SVM.fit(X_train, Y_train)
# predictions_SVM = model_SVM.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_SVM))
# print(confusion_matrix(Y_validation, predictions_SVM))
# print(classification_report(Y_validation, predictions_SVM))

# print("SVM (poly3)" + spec)
# # SVM poly3
# model_SVM = SVC(gamma='auto', kernel='poly', degree=3)
# model_SVM.fit(X_train, Y_train)
# predictions_SVM = model_SVM.predict(X_validation)

# print(accuracy_score(Y_validation, predictions_SVM))
# print(confusion_matrix(Y_validation, predictions_SVM))
# print(classification_report(Y_validation, predictions_SVM))