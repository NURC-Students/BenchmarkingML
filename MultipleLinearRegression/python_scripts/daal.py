import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
import numpy.ma as ma
from numpy import *
from daal4py import daalinit, daalfini, kmeans_init
import daal4py as d4p
import time


print("\nTesting PyDaal...\n")
X = np.array([[1.,1.], [1.,4.], [1.,0.]])
daalinit()
result = kmeans_init(10, method = "plusPlusDense", distributed = True).compute(X)
daalfini()
print(result.centroids)

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
y_float = [y.astype(np.float)]
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
# y_reshaped = np.array(y_float, dtype = np.float32)
print(type(X_float[0]))
print(type(y_float[0]))
print("\nConfiguring an Intel pyDaal Multiple Linear regression training object...\n")
#train_algo = d4p.linear_regression_training(interceptFlag = True)
print("\nTraining the pyDaal Multiple Linear Regression model on the Training set...\n")
start = time.time()
# train_result = train_algo.compute(X_float, y_float)
train_result = d4p.linear_regression_training().compute(X_float, y_float)
print(train_result)
stop = time.time()
print(f"Intel pyDaal Training time: {stop - start} seconds.")
print("Exiting...All Done!")