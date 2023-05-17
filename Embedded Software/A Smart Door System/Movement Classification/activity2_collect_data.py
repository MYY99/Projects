import RPi.GPIO as GPIO
import json
import time
import os
import psutil
import csv
import requests #Please install with PIP: pip install requests
request = None

trigger_delay = 0.2
settling_delay = 0.05
counts = 990

dist_data = []
                
try:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    PIN_TRIGGER = 7
    PIN_ECHO = 11

    GPIO.setup(PIN_TRIGGER, GPIO.OUT)
    GPIO.setup(PIN_ECHO, GPIO.IN)

    GPIO.output(PIN_TRIGGER, GPIO.LOW)
    
    with open('come3.csv','a',encoding='UTF8',newline='') as file:
        
        writer = csv.writer(file)
    
        for count in range(counts):
            # US sensor measures distance
            print ("Waiting for sensor to settle")
            time.sleep(settling_delay)
            print ("Calculating distance")
            GPIO.output(PIN_TRIGGER, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(PIN_TRIGGER, GPIO.LOW)
            while GPIO.input(PIN_ECHO)==0:
                pulse_start_time = time.time()
            while GPIO.input(PIN_ECHO)==1:
                pulse_end_time = time.time()
            pulse_duration = pulse_end_time - pulse_start_time
            distance = round(pulse_duration * 17150, 2)
            print ((count+1) % 30)
            print ("Distance:",distance,"cm")
            print ("Calculating distance")
            time.sleep(trigger_delay)
            
            dist_data.append(distance)
            
            if ((count+1) % 30 == 0):
                writer.writerow(dist_data)
                dist_data = []
                print("sleep 2 sec")
                time.sleep(2)
                

finally:
    GPIO.cleanup()


