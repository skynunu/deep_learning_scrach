import sys
sys.path.append('..')
sys.path.append('../ch7')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq
from ch7.seq2seq import Seq2seq
from ch7.peeky_seq2seq import PeekySeq2seq

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()


# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

"""
# 1. base line 모델
model1 = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer1 = Adam()
trainer1 = Trainer(model1, optimizer1)

acc_list1 = []
for epoch in range(max_epoch):
    trainer1.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model1, question, correct,
                                    id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list1.append(acc)
    print('정확도 %.3f%%' % (acc * 100))

# 2. peeky 모델

# 입력 문장 반전
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

model2 = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer2 = Adam()
trainer2 = Trainer(model2, optimizer2)

acc_list2 = []
for epoch in range(max_epoch):
    trainer2.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model2, question, correct,
                                    id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list2.append(acc)
    print('정확도 %.3f%%' % (acc * 100))
"""

# 3. Attention 모델

# 입력 문장 반전
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

model3 = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer3 = Adam()
trainer3 = Trainer(model3, optimizer3)

acc_list3 = []
for epoch in range(max_epoch):
    trainer3.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model3, question, correct,
                                    id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list3.append(acc)
    print('정확도 %.3f%%' % (acc * 100))
    
model3.save_params()

# 그래프 그리기
x = np.arange(len(acc_list3))
#plt.plot(x, acc_list1, marker='o', label="baseline")
#plt.plot(x, acc_list2, marker="D", label= "peeky")
plt.plot(x, acc_list3, marker="v", label= "attention")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.legend(loc='center right')
plt.show()
