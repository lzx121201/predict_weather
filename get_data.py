from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os

LATITUDE = 53.997945
LONGTITUDE = -6.405957
API_KEY = '3efc4e2b91b740018b1a76bddf7cdbe3'
BASE_URL = "https://api.darksky.net/forecast/{}/{},{},{}?units=si"
START_DATE = int(datetime(2018,2,1).timestamp())



def get_data_streamed_to_local(url,api_key,langtitude,longtitude,target_date,days):
     records = []
     count = 0
     DAY_IN_SECOND = 86400
     for i in range(days):
         request = url.format(API_KEY,langtitude,longtitude, target_date)
         response = requests.get(request)
         data = response.json()
         records.append(data)
         count+=1
         print(count)
         target_date -= DAY_IN_SECOND
     return records

#
records = get_data_streamed_to_local(BASE_URL, API_KEY,LATITUDE,LONGTITUDE ,START_DATE, 1000)
fileName = "C:\\Users\\zhengxing.li\\Desktop\\predict_weather\\data1000.txt"
for record in records:
    file = open(fileName,'a', encoding='utf-8')
    file.write(str(record))
    file.write('\n')
    file.close()
