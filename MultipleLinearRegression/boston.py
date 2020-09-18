import daal4py as d4p
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os

data = load_boston()

X = data.data # house characteristics

y = data.target[np.newaxis].T # house price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1693)
#print(X_train)
print(type(X_train[0]))
print(type(y_train[0]))
train_result = d4p.linear_regression_training().compute(X_train, y_train)
y_pred = d4p.linear_regression_prediction().compute(X_test, train_result.model).prediction 
print("works...")
