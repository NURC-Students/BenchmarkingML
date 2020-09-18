import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy.ma as ma
from numpy import *
import time

print("\nTesting SciKit Learn...\n")
X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)

print("\nReading Dataset...\n")
# dataset = pd.read_csv('/home/s.chakravarty/dataset/aug.csv', engine = 'python', low_memory = True)
# print(dataset.info())
dataset = pd.read_json('/home/s.chakravarty/dataset/jun_jul.json')
print(dataset[0:6])

print("\nSelecting Features...\n")
X = dataset.iloc[1:, [2,3,4,5,6,7,8,11,12,14]].values
X_float = X.astype(np.float)
print(X_float[0:5])

print("\nSelecting Target...\n")
y = dataset.iloc[1:, 9].values
y_float = y.astype(np.float)
print(y_float[0:5])

print("\nDoes Target contain NaN?\n")
array_sum = np.sum(y_float)
array_has_nan = np.isnan(array_sum)
print(array_has_nan)

print("\nCalculating mean for target...\n")
col_mean = np.nanmean(y_float, axis=0)
print(col_mean)

print("\nReplance NaNs with mean value...\n")
where_are_NaNs = isnan(y_float)
y_float[where_are_NaNs] = col_mean

print("\nDoes Target contain any remaining NaNs?\n")
array_sum = np.sum(y_float)
array_has_nan = np.isnan(array_sum)
print(array_has_nan)

print("\nSplitting the dataset into the Training set and Test set...\n")
X_train, X_test, y_train, y_test = train_test_split(X_float, y_float, test_size = 0.2, random_state = 0)

print("\nConfiguring a Scikit-Learn Multiple Linear regression training object...\n")
regressor = LinearRegression()
print(regressor)
print("\nTraining the Scikit-Learn Multiple Linear Regression model on the Training set...\n")
start = time.time()
regressor.fit(X_train, y_train)
stop = time.time()
print(f"Scikit Learn Training time: {stop - start} seconds.")
print("Exiting...All Done!")