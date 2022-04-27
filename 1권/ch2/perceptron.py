import numpy as np

def AND(x1, x2) :
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0 :
        return 0 
    else :
        return 1

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0 :
        return 0 
    else :
        return 1
    
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0 :
        return 0 
    else :
        return 1
    
def XOR(x1, x2) :
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

if __name__ == "__main__":
    print()
    print("AND 게이트 퍼셉트론 가중치,임계값으로 구현")
    print("AND(0,0):", AND(0,0)) 
    print("AND(1,0):", AND(1,0)) 
    print("AND(0,1):", AND(0,1)) 
    print("AND(1,1):", AND(1,1)) 

    print()
    print("NAND 게이트 퍼셉트론 가중치,편향으로 구현")
    print("NAND(0,0):", NAND(0,0)) 
    print("NAND(1,0):", NAND(1,0)) 
    print("NAND(0,1):", NAND(0,1)) 
    print("NAND(1,1):", NAND(1,1)) 

    print()
    print("OR 게이트 퍼셉트론 가중치,편향으로 구현")
    print("OR(0,0):", OR(0,0)) 
    print("OR(1,0):", OR(1,0)) 
    print("OR(0,1):", OR(0,1)) 
    print("OR(1,1):", OR(1,1))
    
    print()
    print("XOR 게이트 다층 퍼셉트론으로 구현")
    print("XOR(0,0):", XOR(0,0)) 
    print("XOR(1,0):", XOR(1,0)) 
    print("XOR(0,1):", XOR(0,1)) 
    print("XOR(1,1):", XOR(1,1))