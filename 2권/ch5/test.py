import numpy as np

ts = np.array([1,2,3,4])
ignore_label = np.array([1,1,1,1])
mask = (ts != ignore_label)
print(mask)

a = [3,3,3,3,3]
for i,v in enumerate(a):
    print(i,v)