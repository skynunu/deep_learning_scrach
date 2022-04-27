import sys, os
from wsgiref.headers import tspecials
sys.path.append(os.pardir) 
from collections import OrderedDict
from common.functions import *
from common.layers import *
import time

input_size = 5
hidden_size = 10
params = {}
params['W1'] =  np.random.randn(input_size, hidden_size)
params['b1'] = np.zeros(hidden_size)


start = time.time() 
a = {}
a['a'] = Affine(params['W1'], params['b1'])
#print(a['a'])
#print(a)

ts = 10
bs = 20
b = np.random.choice(ts, bs)
print(b)

end = time.time() 
print("총학습시간: "+f"{end - start:.5f} sec")

a_dict = {'a':1,'b':2,'c':3}
print("ch6")
print(a_dict)
print(a_dict.keys())
print(a_dict.values())

