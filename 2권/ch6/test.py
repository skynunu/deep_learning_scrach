import numpy as np
a = np.array([1,2,3,4])
b = np.array([1,1,1,1])
A = [a,b]
def add(A) :
    for a in A :
        a +=1 
        print(a)
        
add(A)
print(A)