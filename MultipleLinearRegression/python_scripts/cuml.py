import numpy as np
import cudf

# Both import methods supported
#from cuml import LinearRegression
from cuml.linear_model import LinearRegression

lr = LinearRegression(fit_intercept = True, normalize = False,
                      algorithm = "eig")

X = cudf.DataFrame()
X['col1'] = np.array([1,1,2,2], dtype = np.float32)
X['col2'] = np.array([1,2,2,3], dtype = np.float32)

y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

reg = lr.fit(X,y)
print("Coefficients:")
print(reg.coef_)
print("Intercept:")
print(reg.intercept_)

X_new = cudf.DataFrame()
X_new['col1'] = np.array([3,2], dtype = np.float32)
X_new['col2'] = np.array([5,5], dtype = np.float32)
preds = lr.predict(X_new)

print("Predictions:")
print(preds)
