import sys, os
sys.path.append(os.pardir)  
import numpy as np
import pickle
from dataset.mnist import load_mnist
from activationfunc import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size) :
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

network = init_network()
W1,W2,W3 = network['W1'], network['W2'], network['W3']
print()
print("신경망의  MNIST데이터 추론, batch_size 사용")
print("입력 데이터 shape : ", x.shape)
print("1st layer 가중치 :", W1.shape)
print("2nd layer 가중치 :", W2.shape)
print("3rd layer 가중치 :", W3.shape)
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

