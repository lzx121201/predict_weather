import pandas as pd
import math, random
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import scipy.stats as ss
from copy import deepcopy
import pymongo
from pymongo import MongoClient
from sklearn.cluster import KMeans
import scipy.spatial.distance as sci
from sklearn.decomposition import PCA

#Initialize
studentid =[]
Lab_1 = []
Christmas_Test = []
Lab_2 = []
Easter_Test =[]
Lab_3 = []
part_time_job = []
Exam_Grade = []

def read_in_from_mongo():
    client = MongoClient('localhost', 27017)
    db = client.dbs_final_ca
    collection = db.records
    results = collection.find()
    for result in results:
        studentid.append(int(result["studentid"]))
        Lab_1.append(int(result["Lab_1"]))
        Christmas_Test.append(int(result["Christmas_Test"]))
        Lab_2.append(int(result["Lab_2"]))
        Easter_Test.append(int(result["Easter_Test"]))
        Lab_3.append(int(result["Lab_3"]))
        part_time_job.append(int(result["part_time_job"]))
        Exam_Grade.append(int(result["Exam_Grade"]))

#Read in records from MongoDB
read_in_from_mongo()

#Functions
def mean(x):
    return sum(x)/len(x)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def median(x):
    n = len(x)
    sorted_x = sorted(x)
    midpoint = n //2
    if n % 2 == 1:
        return sorted_x[midpoint]
    else:
        return (sorted_x[midpoint-1]+sorted_x[midpoint]) /2

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()if count == max_count]

def quantile(x, p):
	p_index = int(p * len(x))
	return sorted(x)[p_index]

def variance(x):
    n= len(x)
    avg = mean(x)
    deviations = sum([(i-avg)*(i-avg) for i in x ])
    return deviations/n

def standard_deviation(x):
	return math.sqrt((variance(x)))

def data_range(x):
	return max(x) - min(x)

def interquartile_range(x):
 	return quantile(x, 0.75) - quantile(x, 0.25)

def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n - 1)

from statistics import stdev
def correlation(x, y):
    stdev_x = stdev(x)
    stdev_y = stdev(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0

def check_missing_value(x):
    indexes = []
    for i in range(len(x)):
        if x[i] == 0 or x[i] == "0"or x[i] == None:
            indexes.append(i)
    return indexes

def find_outlier_index(x):
    indexes = []
    lower_inner_fence = quantile(x,0.25)-1.5*interquartile_range(x)
    upper_inner_fence = quantile(x,0.75)+1.5*interquartile_range(x)
    print("lower_inner_fence: ",lower_inner_fence)
    print("upper_inner_fence: ",upper_inner_fence)
    for i in range(len(x)):
        if x[i] < lower_inner_fence or x[i] > upper_inner_fence:
            indexes.append(i)
    return indexes

def partial_difference_quotient(f,v,i,x,y,h):
    w = [v_j +(h if j == i else 0) for j,v_j in enumerate(v)]
    return ( f(w,x,y) - f(v,x,y) )/h

def estimate_gradient_stochastic(f,v,x,y,h=0.01):
    return [partial_difference_quotient(f,v,i,x,y,h) for i,_ in enumerate(v)]

def step_stochastic(v, direction, step_size):
    return [v_i-step_size * direction_i for v_i, direction_i in zip(v,direction)]

def predict_y(v,x):
	return np.dot(v, x)

def error(v,x,y):
	error = y-predict_y(v,x)
	return error

def squared_errors(v,x,y):
	return error(v,x,y)**2

def least_squares_fit(x,y):
	beta = correlation(x,y)*standard_deviation(y)/standard_deviation(x)
	alpha = mean(y)-beta*mean(x)
	return alpha, beta

def stochastic_gradient_descent(x,y,n):
    v = [random.randint(-10,10) for i in range (n)]
    step_0 = 0.000001
    iterations_with_no_improvement = 0
    min_v = v
    min_value = float("inf")
    while iterations_with_no_improvement < 100:
        value = sum(squared_errors(v, x_i, y_i) for x_i, y_i in zip(x,y))
        if value < min_value:
            min_v, min_value = v,value
            iterations_with_no_improvement = 0
            step_size = step_0
        else:
            iterations_with_no_improvement += 1
            step_size *= 0.9
        indexes = np.random.permutation(len(x))
        for i in indexes:
            x_i = x[i]
            y_i = y[i]
            gradient_i=estimate_gradient_stochastic(squared_errors,v,x_i, y_i)
            v = step_stochastic(v, gradient_i, step_size)
        # print(min_v)
    return min_v

def sum_of_squares(x):
	return sum([x_i**2 for x_i in x])

def multiple_r_squared(v, x, y):
	sum_of_squared_errors = sum(error(v, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))
	return 1.0 - sum_of_squared_errors / sum_of_squares(de_mean(y))

def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


print("- Missing value: ")
print("Lab_1: ",check_missing_value(Lab_1))
print("Christmas_Test: ",check_missing_value(Christmas_Test))
print("Lab_2: ",check_missing_value(Lab_2))
print("Easter_Test: ",check_missing_value(Easter_Test))
print("Lab_3: ",check_missing_value(Lab_3))
print("part_time_job: ",[i for i in part_time_job if i !=1 and i !=0])
print("Exam_Grade: ",check_missing_value(Exam_Grade))
print()

print("- Outliers value: ")
print("Lab_1: ",[Lab_1[i] for i in find_outlier_index(Lab_1)])
print()
print("Christmas_Test: ",[Christmas_Test[i] for i in find_outlier_index(Christmas_Test)])
print()

print("Lab_2: ",[Lab_2[i] for i in find_outlier_index(Lab_2)])
print()

print("Easter_Test: ",[Easter_Test[i] for i in find_outlier_index(Easter_Test)])
print()

print("Lab_3: ",[Lab_3[i] for i in find_outlier_index(Lab_3)])
print()

print("Exam_Grade: ",[Exam_Grade[i] for i in find_outlier_index(Exam_Grade)])
print()

decile = lambda x: x //10*10
Lab_1_histogram = Counter( decile (i) for i in Lab_1)
plt.xticks([10*i for i in range(11)])
plt.bar([x-4 for x in Lab_1_histogram.keys()], Lab_1_histogram.values(), 9)
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of Lab 1 Grades")
plt.yticks([x*50 for x in range(11)])
plt.xlim(-5,105)
plt.show()

Christmas_Test_histogram = Counter( decile (i) for i in Christmas_Test)
plt.xticks([10*i for i in range(11)])
plt.bar([x-4 for x in Christmas_Test_histogram.keys()], Christmas_Test_histogram.values(), 9)
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of Christmas Test Grades")
plt.yticks([x*50 for x in range(11)])
plt.xlim(-5,105)
plt.show()

Lab_2_histogram = Counter( decile (i) for i in Lab_2)
plt.xticks([10*i for i in range(11)])
plt.bar([x-4 for x in Lab_2_histogram.keys()], Lab_2_histogram.values(), 9)
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of Lab 2 Grades")
plt.yticks([x*50 for x in range(11)])
plt.xlim(-5,105)
plt.show()

Easter_Test_histogram = Counter( decile (i) for i in Easter_Test)
plt.xticks([10*i for i in range(11)])
plt.bar([x-4 for x in Easter_Test_histogram.keys()], Easter_Test_histogram.values(), 9)
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of Easter Test Grades")
plt.yticks([x*50 for x in range(11)])
plt.xlim(-5,105)
plt.show()

Lab_3_histogram = Counter( decile (i) for i in Lab_3)
plt.xticks([10*i for i in range(11)])
plt.bar([x-4 for x in Lab_3_histogram.keys()], Lab_3_histogram.values(), 9)
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of Lab 3 Grades")
plt.yticks([x*50 for x in range(11)])
plt.xlim(-5,105)
plt.show()

part_time_job_histogram = Counter(part_time_job)
plt.bar(part_time_job_histogram.keys(),part_time_job_histogram.values())
plt.xticks([i for i in range(0,2)])
plt.yticks([x*50 for x in range(13)])
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of students who have/don't have part_time_job")
plt.show()

Exam_Grade_histogram = Counter( decile (i) for i in Exam_Grade)
plt.xticks([10*i for i in range(11)])
plt.bar([x-4 for x in Exam_Grade_histogram.keys()], Exam_Grade_histogram.values(), 9)
plt.xlabel("Grades")
plt.ylabel("# of students")
plt.title("Distribution of Exam Grades")
plt.yticks([x*50 for x in range(13)])
plt.xlim(-5,105)
plt.show()

print("Lab_1 & Exam_Grade")
print("covariance: ",covariance(Lab_1,Exam_Grade))
print("correlation: ",correlation(Lab_1,Exam_Grade))
print("R^2: ",correlation(Lab_1,Exam_Grade)**2)
intercept ,slope = least_squares_fit(Lab_1,Exam_Grade)
print("slope: ",slope)
print("intercept: ",intercept)

print("\nChristmas_Test & Exam_Grade")
print("covariance: ",covariance(Christmas_Test,Exam_Grade))
print("correlation: ",correlation(Christmas_Test,Exam_Grade))
print("R^2: ",correlation(Christmas_Test,Exam_Grade)**2)
intercept1 ,slope1 = least_squares_fit(Christmas_Test,Exam_Grade)
print("slope: ",slope1)
print("intercept: ",intercept1)

print("\nLab_2 & Exam_Grade")
print("covariance: ",covariance(Lab_2,Exam_Grade))
print("correlation: ",correlation(Lab_2,Exam_Grade))
print("R^2: ",correlation(Lab_2,Exam_Grade)**2)
intercept2 ,slope2 = least_squares_fit(Lab_2,Exam_Grade)
print("slope: ",slope2)
print("intercept: ",intercept2)

print("\nEaster_Test & Exam_Grade")
print("covariance: ",covariance(Easter_Test,Exam_Grade))
print("correlation: ",correlation(Easter_Test,Exam_Grade))
print("R^2: ",correlation(Easter_Test,Exam_Grade)**2)
intercept3 ,slope3 = least_squares_fit(Easter_Test,Exam_Grade)
print("slope: ",slope3)
print("intercept: ",intercept3)
# print("Simple linear regression model: y = ",intercept3," + ",slope3,"x")
#
# plt.scatter(Easter_Test,Exam_Grade)
# axes = plt.gca()
# m, b = np.polyfit(Easter_Test,Exam_Grade, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b,'g-')
# plt.xlabel("Easter_Test")
# plt.ylabel("Exam_Grade")
# plt.xticks([x*10 for x in range(11)])
# plt.yticks([x*10 for x in range(11)])
# plt.title("Easter_Test & Exam_Grade")
# plt.show()
#
print("\nLab_3 & Exam_Grade")
print("covariance: ",covariance(Lab_3,Exam_Grade))
print("correlation: ",correlation(Lab_3,Exam_Grade))
print("R^2: ",correlation(Lab_3,Exam_Grade)**2)
intercept4 ,slope4 = least_squares_fit(Lab_3,Exam_Grade)
print("slope: ",slope4)
print("intercept: ",intercept4)

#Multiple linear Regression
# print("\nMultiple linear Regression")
# x = [[1,Lab_1[i],Christmas_Test[i],Lab_2[i],Easter_Test[i],Lab_3[i]] for i in range(len(studentid))]
# y = Exam_Grade
# estimate_v = stochastic_gradient_descent(x,y,len(x[0]))
# print("estimate_v: ",estimate_v)
#
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
#

# K-means Clusters Analysis
x = [[Lab_1[i],Christmas_Test[i],Lab_2[i],Easter_Test[i],Lab_3[i],part_time_job[i],part_time_job[i],Exam_Grade[i]] for i in range(len(studentid))]

no_clustering = range(1,11)
average_dist = []
for k in no_clustering:
    model_k = KMeans(k)
    model_k .fit(x)
    cluster_assign  = model_k.predict(x)
    average_dist.append(sum(np.min(sci.cdist(x,model_k.cluster_centers_,'euclidean'),axis =1))/len(x))

plt.plot(no_clustering, average_dist, 'bx-')
plt.xlabel('k')
plt.ylabel('average_dist')
plt.title('The Elbow Method showing the optimal k')
plt.xticks([x for x in range(1,11)])
plt.show()


#Principal Component Analysis
pca =PCA()
plot_data = pca.fit_transform(x)
PC = pca.components_
print("Principal Component: ",PC)
PCEV=pca.explained_variance_
PCEVR=pca.explained_variance_ratio_
model_k = KMeans(2)
model_k .fit(x)
center = model_k.cluster_centers_
print("center: ",center)
a=[i+1 for i in range(len(PC))]
plt.plot(a, PCEVR)
plt.xlabel('Principal Component')
plt.ylabel('Variation Explained')
plt.title('Scree-plot')
plt.show()

pca =PCA(2)
plot_data = pca.fit_transform(x)
PC = pca.components_
print("Principal Component: ",PC)

plt.scatter(x=plot_data[:,0], y=plot_data[:,1], c=model_k.labels_,)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatterplot of Principal Components for 2 Clusters')
plt.show()

model_k = KMeans(3)
model_k .fit(x)
center = model_k.cluster_centers_
print("center: ",center)
plt.scatter(x=plot_data[:,0], y=plot_data[:,1], c=model_k.labels_,)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatterplot of Principal Components for 3 Clusters')
plt.show()
