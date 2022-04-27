import numpy as np


dW = np.array([[1,2,3,4],[3,4,2,5]])
dout = np.array([1,1,1,1])
dW[...] = 0
print(dW.shape)
print(dW)

np.add.at(dW, 1, dout)
print(dW)