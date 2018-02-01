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


features = ["date", "dewpointMean", "dewpointMax", "dewpointMin", "humidityMean", "humidityMax", "humidityMin",
            "temperatureMean", "temperatureMax", "temperatureMin", "pressureMean", "pressureMax", "pressureMin"]
DailySummary = namedtuple("DailySummary", features)
data = []
fileName = "C:\\Users\\zhengxing.li\\Desktop\\predict_weather\\data500.txt"
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

predict_feature = [x for x in predict_feature if x != 'temperatureMean_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())


predict_feature = [x for x in predict_feature if x != 'dewpointMax_2']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMean_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMax_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMin_2']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMean_2']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMin_2']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'dewpointMin_1']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

predict_feature = [x for x in predict_feature if x != 'temperatureMin_1']
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

predict_feature = [x for x in predict_feature if x != 'temperatureMax_3']
x = sm.add_constant(df[predict_feature])
model = sm.OLS(y, x).fit()
# print(model.summary())

x = x.drop('const', axis=1)
model = sm.OLS(y, x).fit()
print(model.summary())


lrm = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=12)

lrm.fit(X_train, y_train)

print("R^2: ", lrm.score(X_test, y_test))

print(lrm.coef_)
print(predict_feature)

predict_x = df[predict_feature].head(1)
predict_y = lrm.predict(predict_x)
print(predict_y)


# x = x.values
# x = [np.append(x[i], 1) for i in range(len(x))]


def mean(x):
    return sum(x) / len(x)


def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


def partial_difference_quotient(f, v, i, x, y, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w, x, y) - f(v, x, y)) / h


def estimate_gradient_stochastic(f, v, x, y, h=0.01):
    return [partial_difference_quotient(f, v, i, x, y, h) for i, _ in enumerate(v)]


def step_stochastic(v, direction, step_size):
    return [v_i - step_size * direction_i for v_i, direction_i in zip(v, direction)]


def predict_y(v, x):
    return np.dot(v, x)


def error(v, x, y):
    error = y - predict_y(v, x)
    return error


def squared_errors(v, x, y):
    return error(v, x, y)**2


def least_squares_fit(x, y):
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def stochastic_gradient_descent(x, y, n):
    v = [random.randint(-10, 10) for i in range(n)]
    step_0 = 0.000001
    iterations_with_no_improvement = 0
    min_v = v
    min_value = float("inf")
    while iterations_with_no_improvement < 100:
        value = sum(squared_errors(v, x_i, y_i) for x_i, y_i in zip(x, y))
        if value < min_value:
            min_v, min_value = v, value
            iterations_with_no_improvement = 0
            step_size = step_0
        else:
            iterations_with_no_improvement += 1
            step_size *= 0.9
        indexes = np.random.permutation(len(x))
        for i in indexes:
            x_i = x[i]
            y_i = y[i]
            gradient_i = estimate_gradient_stochastic(
                squared_errors, v, x_i, y_i)
            v = step_stochastic(v, gradient_i, step_size)
        print(min_v)
    return min_v


def sum_of_squares(x):
    return sum([x_i**2 for x_i in x])


def multiple_r_squared(v, x, y):
    sum_of_squared_errors = sum(
        error(v, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / sum_of_squares(de_mean(y))


def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


estimate_v = [-0.021811667133340939, 0.14687803265007829, 0.92650921181033641, -1.5637082888822083, -
              0.79940496458061017, 1.4707604100907701, 0.34925735908857447, 0.60861098983278528, -3.5726569393287502]

# estimate_v = stochastic_gradient_descent(x,y,len(x[0]))
# print("estimate_v: ",estimate_v)

#
# def bootstrap_statistic_m(x,y, stats_fn, num_samples):
#     stats = []
#     for i in range(num_samples):
#         data = zip(x,y)
#         sample_data = bootstrap_sample(data)
#         stat=stats_fn(sample_data)
#         stats.append(stat)
#     return stats
#
# def bootstrap_sample(data):
#     list_data = list(data)
#     rand_data = [random.choice(list_data) for _ in list_data]
#     return rand_data
#
# def estimate_sample_v(sample):
#     x_sample, y_sample = zip(*sample)
#     return stochastic_gradient_descent(x_sample,y_sample,len(x_sample[0]))
#

# coefficients = bootstrap_statistic_m(x,y,estimate_sample_v,20)
# # print("boot_strap x,y",coefficients)
#
#
#
# bootstrap_standard_errors = [standard_deviation([coefficient[i] for coefficient in coefficients]) for i in range(len(estimate_v))]
#
# # print("bootstrap_standard_errors: ",bootstrap_standard_errors)
#
# print("R^2: ",multiple_r_squared(estimate_v,x,y))
#
# for i in range(len(estimate_v)):
#  	print("i",i,"estimate_v",estimate_v[i],"error", bootstrap_standard_errors[i],"p-value", p_value(estimate_v[i], bootstrap_standard_errors[i]))
