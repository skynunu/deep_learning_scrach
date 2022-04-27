import numpy as np

def function_2(x) : 
    return x[0]**2 + x[1]**2
    

def numerical_gradient(f,x) :
    h = 1e-4
    grad = np.zeros_like(x) #x와 형상이 같은 배열 생성

    for idx in range(x.size) :
        tmp_val = x[idx]

        #f(x+h) 계산
        x[idx] = tmp_val+h
        fxh1 = f(x)

        #f(x-h) 계산
        x[idx] = tmp_val-h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100) :
    x = init_x

    for i in range(step_num) :
        grad = numerical_gradient(f,x)
        x -= lr*grad

    return x

if __name__ == "__main__": 
    print("x[0]**2 + x[1]**2에 대한 기울기 계산")
    print("3과 4에 대한 기울기 :",numerical_gradient(function_2, np.array([3.0, 4.0])))
    print("0과 2에 대한 기울기 :",numerical_gradient(function_2, np.array([0.0, 2.0])))
    print("3과 0에 대한 기울기 :",numerical_gradient(function_2, np.array([3.0, 0.0])))


    init_x = np.array([-3.0, 4.0])
    print("학습율에 따른 경사하강법 비교")
    print("초기값 :-3,4 step:100")
    print("학습율:0.1 =>",gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
    print("학습율 클때 10.0: =>",gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
    print("학습율 작을때:1e-10 =>",gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
