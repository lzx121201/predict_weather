from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import math
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

def read_data(fileName):
    with open(fileName, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            data.append(line)


def convert_data_to_dict():
    for i in range(0, len(data)):
        dict.append(ast.literal_eval(data[i]))


def extract_data(dict, namedtuple):
    for line in dict:
        hourlyData = line['hourly']['data']
        dailyData = line['daily']['data'][0]

        date = datetime.fromtimestamp(dailyData['time']).strftime('%Y-%m-%d')
        dewpoints = [i['dewPoint'] for i in hourlyData]
        dewpointMean = dailyData['dewPoint']
        dewpointMax = max(dewpoints)
        dewpointMin = min(dewpoints)

        humiditys = [i['humidity'] for i in hourlyData]
        humidityMean = dailyData['humidity']
        humidityMax = max(humiditys)
        humidityMin = min(humiditys)

        pressures = [i.get('pressure', 0) for i in hourlyData]
        pressureMean = dailyData['pressure']
        pressureMax = max(pressures)
        pressureMin = min(pressures)

        temperatures = [i['temperature'] for i in hourlyData]
        totalTemparature = sum(temperatures)
        temperatureMean = sum(temperatures) / len(temperatures)
        temperatureMax = max(temperatures)
        temperatureMin = min(temperatures)

        records.append(DailySummary(
            date=date,
            dewpointMean=dewpointMean,
            dewpointMax=dewpointMax,
            dewpointMin=dewpointMin,
            humidityMean=humidityMean,
            humidityMax=humidityMax,
            humidityMin=humidityMin,
            pressureMean=pressureMean,
            pressureMax=pressureMax,
            pressureMin=pressureMin,
            temperatureMean=temperatureMean,
            temperatureMax=temperatureMax,
            temperatureMin=temperatureMin
        ))


def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None] * N + \
        [df[feature][i - N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

def replace_nan(df):
    na_columns = df.columns[df.isnull().any()].tolist()
    for column in na_columns:
        df[column] = df[column].fillna(df[column].mean())

features = ["date", "dewpointMean", "dewpointMax", "dewpointMin", "humidityMean", "humidityMax", "humidityMin",
            "temperatureMean", "temperatureMax", "temperatureMin", "pressureMean", "pressureMax", "pressureMin"]
DailySummary = namedtuple("DailySummary", features)
data = []
fileName = "C:\\Users\\zhengxing.li\\Desktop\\predict_weather\\data1000.txt"
dict = []
records = []

read_data(fileName)
convert_data_to_dict()
extract_data(dict, DailySummary)

df = pd.DataFrame(records, columns=features).set_index('date')

for feature in features:
    if feature != 'date':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)

replace_nan(df)

to_remove = [feature
             for feature in features
             if feature not in ['temperatureMean', 'temperatureMin', 'temperatureMax']]

to_keep = [col for col in df.columns if col not in to_remove]
df = df[to_keep]

spread = df.describe().T

df = df.dropna()

corrs = df.corr()[['temperatureMean']].sort_values('temperatureMean')

indexes = corrs.index.get_values()
values = corrs['temperatureMean'].values

predict_feature = []
for i in range(len(indexes)):
    if abs(values[i]) >= 0.6 and abs(values[i]) < 1:
        predict_feature.append(indexes[i])

x_0 = df[predict_feature]
y = df['temperatureMean']

x = sm.add_constant(x_0)
alpha = 0.05

model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMean_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMin_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMin_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMean_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMax_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMean_2']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMax_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

x = df[predict_feature]
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMin_2']
x = df[predict_feature]
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMax_2']
x = df[predict_feature]
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMin_2']
x = df[predict_feature]
model = sm.OLS(y, x).fit()
# print(model.summary())


lrm = linear_model.LinearRegression()
validation_size = 0.20
seed = 14
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=validation_size, random_state=seed)

lrm.fit(X_train, y_train)

print("Train set R^2: ", lrm.score(X_train, y_train))
print("Test set R^2: ", lrm.score(X_test, y_test))

print("coefficients: ",lrm.coef_)
print("predictor: ",predict_feature)

prediction = lrm.predict(X_test)
print("The Explained Variance: %.2f" % lrm.score(X_test, y_test))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))


data = []
fileName = "C:\\Users\\zhengxing.li\\Desktop\\predict_weather\\data_1Feb2018.txt"
dict = []
records = []

read_data(fileName)
convert_data_to_dict()
extract_data(dict, DailySummary)

df = pd.DataFrame(records, columns=features).set_index('date')

for feature in features:
    if feature != 'date':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)

replace_nan(df)

predict_x = df[predict_feature].head(1)
predict_y = lrm.predict(predict_x)
print(predict_x)
print("Mean temperature of 2 Feb 2018: ",predict_y)
