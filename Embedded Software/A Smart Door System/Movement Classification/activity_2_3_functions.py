# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:34:05 2022

@author: User
"""
import RPi.GPIO as GPIO
import json
import time
import os
import psutil
import csv
import requests #Please install with PIP: pip install requests
request = None

from pandas import read_csv
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import joblib
import math
from datetime import datetime

import Adafruit_DHT
import statistics as s

model_MLP_bin = joblib.load('model_MLP_bin.sav')
model_MLP_dyn = joblib.load('model_MLP_dyn.sav')
model_CARTS_weather = joblib.load('model_CARTS_weather.sav')

def clean_data(data, save_csv = None, fname_clean = 'cleaned.csv', fname_raw = 'raw.csv'):
    """clean dist. data
    :param data: dist. data to clean, numpy array of shape (num_sample, num_feature)
    :param save_csv: output csv, Boolean <True or False>
    :param filename: csv filename, string <filename.csv>
    :return cleaned data, number of cleaned samples
    """
    if save_csv:
        np.savetxt(fname_raw, data, delimiter=",")
        # np.savetxt(fname_clean, X, delimiter=",")
    
    X = data
    num_sample, num_fea = X.shape
    cleaned = 0
    
    for i in range(num_sample):
        if (X[i,:] > 200).any():
            cleaned += 1
            # print('before: ', i, X[i, :])
            for j in range(num_fea):
                if X[i, j] > 200:
                    if j == 0:
                        if X[i, j+1] > 200 and X[i, j+2] > 200:
                            X[i, j] = 200
                        else:
                            X[i, j] = np.min([X[i, j+1], X[i, j+2]])
                    elif (j > 0) and (j < (num_fea - 1)):
                        if X[i, j+1] > 200:
                            if j != num_fea - 2 and X[i, j+2] > 200:
                                X[i, j] = 200
                            else:
                                X[i, j] = X[i, j-1]
                        else:
                            X[i, j] = np.mean([X[i, j-1], X[i, j+1]])
                    elif j == num_fea - 1:
                        X[i, j] = X[i, j-1]
            # print('after: ', i, X[i, :])
            
    if save_csv:
        # np.savetxt(fname_raw, data, delimiter=",")
        np.savetxt(fname_clean, X, delimiter=",")
        
    return X #, cleaned

def classify_DnC(dist_arr):
    pred_movement = model_MLP_bin.predict(dist_arr)
    if pred_movement[0] == 'static':
        if np.mean(dist_arr) < 50:
            pred_movement[0] = 'near stationary'
        elif np.mean(dist_arr) < 100:
            pred_movement[0] = 'far stationary'
        else:
            pred_movement[0] = 'no stationary'
    else:
        pred_movement = model_MLP_dyn.predict(dist_arr)
    return pred_movement

def send_alert(payload_str,payload_str2=None,temp_humidity=None):
    print(payload_str)
    r = requests.post("https://maker.ifttt.com/trigger/movement_info/with/key/9RG1SHtkUIDoTZFaC5r0A",
                  json = {"value1":payload_str, "value2":payload_str2, "value3": temp_humidity})
    if r.status_code == 200:
        print("Alert Sent")
    else:
        print("Error")
        
def dew_point_txt(avg_temp, avg_humidity, mode=1):
    dew_point = round(avg_temp - ((100 - avg_humidity)/5), 2)
    
    env_status = 'Dew point={0:0.1f}*C<br>'.format(dew_point)
    env_status += 'Status=<br>'
    
    if (mode == 1):
        if dew_point > 26:
            env_status += 'Severely uncomfortable, highly oppresive'
        elif dew_point > 24 and dew_point <= 26:
            env_status += 'Extremely uncomfortable, fairly oppresive'
        elif dew_point > 21 and dew_point <= 24:
            env_status += 'Very humid, quite uncomfortable'
        elif dew_point > 18 and dew_point <= 21:
            env_status += 'Somewhat uncomfortable for most people at upper edge'
        elif dew_point > 16 and dew_point <= 18:
            env_status += 'Okay for most'
        elif dew_point > 13 and dew_point <= 16:
            env_status += 'Comfortable'
        elif dew_point > 10 and dew_point <= 13:
            env_status += 'Very comfortable'
        elif dew_point <= 10:
            env_status += 'Air could be a bit dry for some'
        else:
            env_status += 'Inconclusive'
    
    elif (mode == 0):
        if avg_temp >= 22.5 and avg_temp <= 25.5 and avg_humidity <= 70:
            env_status += 'Comfortable'
        else:
            if avg_temp <= 22.5:
                env_status += 'Low temp. '
            elif avg_temp >= 25.5:
                env_status += 'High temp. '
            elif avg_humidity > 70:
                env_status += 'High humidity. '
            env_status += 'May lead to discomfort.'        
            
    else:
        env_status += 'User comment mode undefined'
    
    return env_status

def temp_humidity_info(temperature_list, humidity_list, time_list, num_sample):
    avg_temp = s.mean(temperature_list[-num_sample:])
    avg_humidity = s.mean(humidity_list[-num_sample:])
    time_str = '<br><br>During ' + time_list[0] + ' to ' + datetime.now().strftime("%H:%M:%S") + '<br>'
    weather = model_CARTS_weather.predict(np.array([avg_temp, avg_humidity]).reshape(1, -1))[0]
    weather_str = 'Weather=' + weather + '<br>'
    temp_humidity = time_str +\
                    'Temp={0:0.1f}*C<br>\
                    Humidity={1:0.1f}%<br>'.format(avg_temp, avg_humidity) + \
                    weather_str + \
                    dew_point_txt(avg_temp, avg_humidity)
    return temp_humidity
                            