from daal4py import kmeans_init
import numpy as np

X = np.array([[1.,1.], [1.,4.], [1.,0.]])
kmi = kmeans_init(10, method = "plusPlusDense")
result = kmi.compute(X)
print(result.centroids)

