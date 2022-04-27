import numpy as np

def sum_squares_eror(y,t) :
    return 0.5 * np.sum((y-t)**2)

def cross_etropy_error_basic(y,t):
    epsilon = 1e-7
    return -np.sum(t*np.log(y+epsilon))

def cross_etropy_error(y,t): # 정답 레이블이 2나 7등으 숫자레이블로 주어졌을때
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]

    # 정답 레이블이 2나 7등으 숫자레이블로 주어졌을때
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

    # 정답레이블이 원핫 인코딩으로 주어졌을때
    # return -np.sum(t*np.log(y+1e-7))/batch_size


if __name__ == "__main__":
    #MSE
    print("MSE")
    t = [0,0,1,0,0,0,0,0,0,0]# 정답은 '2'

    # 예1 : '2'일 확률이 가장 높다고 추정함(0.6)
    y1 = [ 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print("예1의 MSE 오차 :",sum_squares_eror(np.array(y1), np.array(t)))

    # 예2 : '7'일 확률이 가장 높다고 추정함(0.6)
    y2 = [ 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print("예2의 MSE 오차 :",sum_squares_eror(np.array(y2), np.array(t)))

    #cross entropy
    print("cross-entropy error")
    t = [0,0,1,0,0,0,0,0,0,0]# 정답은 '2'

    # 예1 : '2'일 확률이 가장 높다고 추정함(0.6)
    y1 = [ 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print("예1의 cross-entropy 오차 :",cross_etropy_error_basic(np.array(y1), np.array(t)))

    # 예2 : '7'일 확률이 가장 높다고 추정함(0.6)
    y2 = [ 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print("예2의 cross-entropy 오차 :",cross_etropy_error_basic(np.array(y2), np.array(t)))



