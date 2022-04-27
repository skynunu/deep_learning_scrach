# 활성화 함수 
import numpy as np
import matplotlib.pyplot as plt

def step_function(x) : 
    y = x > 0 
    return y.astype(np.int)

def sigmoid(x) : 
    return 1/(1+np.exp(-x))

def relu(x) : 
    return np.maximum(0,x)

def identitiy_function(x): # 항등 함수
    return x

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def softmax_c(a) : #오버플로를 막는 소프트맥스함수
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def tanh(x) :
    return np.tanh(x)

if __name__ == "__main__":
    
    #계단함수 그래프 출력
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x,y)
    plt.ylim(-0.1, 1.1)
    plt.show()

    #시그모이드 함수 그래프 출력
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x,y)
    plt.ylim(-0.1, 1.1)
    plt.show()

    #ReLU함수 그래프 출력
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x,y)
    plt.ylim(-0.1, 1.1)
    plt.show()


    #ReLU함수 그래프 출력
    x = np.arange(-5.0, 5.0, 0.1)
    y = tanh(x)
    plt.plot(x,y)
    plt.ylim(-1.1, 1.1)
    plt.show()


    #softmax함수 결과 출력
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print()
    print("0.3, 2.9, 4.0에 대해 softmax적용 : ", y)
    print(y, "의 합 : ", np.sum(y))
    
