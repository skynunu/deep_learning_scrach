# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common import config
from common.util import to_cpu, to_gpu
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer
import time 
config.GPU = False

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

if config.GPU :
    x_train, t_train = to_gpu(x_train), to_gpu(t_train)


max_epochs =20
network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)

start = time.time() 
print("start time :",start)

trainer.train()

end = time.time()
print("end time : ", end)
print(f"{end - start:.4f} sec")


# 매개변수 보관
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")


test_acc = network.accuracy(x_test, t_test)
train_acc = network.accuracy(x_train, t_train)

print("================= Final Accuracy =================")
print("test acc:" + str(test_acc))
print("train acc:" + str(train_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()