from daal4py import daalinit, daalfini, kmeans_init
import numpy as np

X = np.array([[1.,1.], [1.,4.], [1.,0.]])
daalinit()
result = kmeans_init(10, method="plusPlusDense", distributed=True).compute(X)
daalfini()
print(result.centroids)