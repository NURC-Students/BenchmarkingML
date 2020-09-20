import cudf
import cupy as cp
from cuml.cluster import KMeans
from cuml.preprocessing.model_selection import train_test_split
from cuml.linear_model import LinearRegression
from cuml.metrics.regression import r2_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import numpy as np
import numpy.ma as ma
from numpy import *

import pandas as pd

import matplotlib.pyplot as plt

import sys
