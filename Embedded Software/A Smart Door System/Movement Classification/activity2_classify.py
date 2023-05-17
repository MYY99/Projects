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
# from sklearn.tree import DecisionTreeClassifier
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

from activity_2_3_functions import *

trigger_delay = 0.2
settling_delay = 0.05
output_delay = 0.00001

num_fea = 30
# counts = 12000
count = 0

tele_start_hr = 0
tele_period_hr = 1
tele_period_min = 2
tele_start_min = datetime.now().minute + tele_period_min + 1

dist_list = []
movement_list = []
movement_list2 = []

time_list = []
time_list2 = []

temperature_list = []
humidity_list = []
num_high_temp = 3
temp_threshold = 58 #20

payload_str = "<br>"
payload_str2 = "<br>"

intermit = ""
intermit2 = ""

abnormal_min = 2
abnormal_count = math.floor(abnormal_min * 60 / 8)
# print(abnormal_count)

abnormal_event_message = "ABNORMAL EVENT DETECTION!!! Object detected in front of the door for more than " + str(abnormal_min) + " minutes!!!<br>"
abnormal_high_temp_message = "Abnormally high temperature detected!!!<br>"


def movement_info(payload_str_local, intermit_local, movement_list_local, time_list_local, event="Movement Info<br>", flag=0, temp_humidity=None
                  ):

    global tele_start_min
    global tele_period_min
    # global tele_start_hr
    # global tele_period_hr
#     print(time_list_local)
#     print(movement_list_local)
    payload_str_local += time_list_local[0]
    
    for i in range(1, len(movement_list_local)):
        if (movement_list_local[i] != movement_list_local[i-1]):
            payload_str_local += " to " + time_list_local[i] + "<br>" + movement_list_local[i-1] +"<br>" + time_list_local[i]
        if i == len(movement_list_local) - 1:
            intermit_local = datetime.now().strftime("%H:%M:%S")
            payload_str_local += " to " + intermit_local + "<br>" + movement_list_local[i]
    
    print(temp_humidity)
    send_alert(payload_str_local, event, temp_humidity)
    
    if flag == 0:
        movement_list_local = []
        time_list_local = []
        tele_start_min = (tele_start_min + tele_period_min) % 60
#         tele_start_hr = (tele_start_hr + tele_period_hr) % 24
        time_list_local.append(intermit_local)

    payload_str_local = "<br>" + datetime.now().strftime("%d %b %Y") + "<br>"
        
    return payload_str_local, intermit_local, movement_list_local, time_list_local

try:
    ### GPIO setting for US sensor
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    PIN_TRIGGER = 7
    PIN_ECHO = 11

    GPIO.setup(PIN_TRIGGER, GPIO.OUT)
    GPIO.setup(PIN_ECHO, GPIO.IN)

    GPIO.output(PIN_TRIGGER, GPIO.LOW)
    
    ### GPIO setting for temperature and humidity sensor
    sensor = Adafruit_DHT.AM2302
    GPIO_AM2302 = 18
    
    ### US sensor initialization
    print("Wait until second = 0...")
    while datetime.now().second != 0:
        None
    
    intermit = datetime.now().strftime("%H:%M:%S")
    intermit2 = datetime.now().strftime("%H:%M:%S")
    
#     for count in range(counts):
    while (1):
        ### MOVEMENT
        if (count % num_fea == 0):
            if payload_str == '<br>':
                payload_str += datetime.now().strftime("%d %b %Y") + '<br>'
                payload_str2 += datetime.now().strftime("%d %b %Y") + '<br>'
                time_list.append(intermit)
                time_list2.append(intermit2)

            else:
                time_list.append(datetime.now().strftime("%H:%M:%S"))
                time_list2.append(datetime.now().strftime("%H:%M:%S"))
                print(datetime.now().strftime("%H:%M:%S"))
        
        # US sensor measures distance
        #print ("Waiting for sensor to settle")
        time.sleep(settling_delay)
        #print ("Calculating distance")
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(PIN_TRIGGER, GPIO.LOW)
        while GPIO.input(PIN_ECHO)==0:
            pulse_start_time = time.time()
        while GPIO.input(PIN_ECHO)==1:
            pulse_end_time = time.time()
        pulse_duration = pulse_end_time - pulse_start_time
        distance = round(pulse_duration * 17150, 2)
        time.sleep(trigger_delay)
        
        dist_list.append(distance)
        
#         if count == 35: temperature_threshold = 20
#         if count == 200: temperature_threshold = 70
        
        ### MOVEMENT CLASSIFICATION
        if ((count+1) % num_fea == 0):            
            dist_arr = np.array(dist_list).reshape(1, -1)
            dist_arr = clean_data(dist_arr)
            pred_movement = classify_DnC(dist_arr)
            
            print("dist: ", dist_arr)
            print("movement: ", pred_movement, "\n")
            
            movement_list.append(pred_movement[0])
            movement_list2.append(pred_movement[0])
            dist_list = []
            
            ### HUMIDITY AND TEMPERATURE
            humidity, temperature = Adafruit_DHT.read_retry(sensor, GPIO_AM2302)
            humidity_list.append(humidity)
            temperature_list.append(temperature)
            print('Temp={0:0.1f}*C  Humidity={1:0.1f}%'.format(temperature, humidity))
            if (len(temperature_list) >= num_high_temp) and any(temperature_list) and (min(temp for temp in temperature_list[-num_high_temp:] if temp is not None) > temp_threshold):
                print("high temp")
                temp_humidity = temp_humidity_info(temperature_list, humidity_list, time_list[-num_high_temp:], num_high_temp)
                send_alert(temp_humidity, abnormal_high_temp_message)
            
#         if datetime.now().hour == tele_start_hr:
        
        ### NORMAL MOVEMENT INFO
#         print(movement_list)
        if datetime.now().minute == tele_start_min:
            temp_humidity = temp_humidity_info(temperature_list, humidity_list, time_list, num_fea)
            payload_str, intermit, movement_list, time_list = movement_info(payload_str, intermit, movement_list, time_list, temp_humidity=temp_humidity)
            print("normal\n")
        
        ### ABNORMAL EVENT DETECTION
#         print(movement_list2)
#         print(math.floor(abnormal_count / 6))
        if (len(movement_list2) == abnormal_count):
            if movement_list2.count("no stat") <= math.ceil(abnormal_count / 6):
                temp_humidity = temp_humidity_info(temperature_list, humidity_list, time_list2, len(movement_list2))
                _, intermit2, movement_list2, time_list2 = movement_info(payload_str2, intermit2, movement_list2, time_list2, event=abnormal_event_message, flag=1, temp_humidity=temp_humidity)
                print("abnormal\n")
            
            movement_list2.pop(0)
            time_list2.pop(0)
        
#         print(movement_list2)
        count += 1

finally:
    GPIO.cleanup()


                                                                                    