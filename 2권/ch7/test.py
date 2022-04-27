import numpy as np
import numpy as np
import matplotlib.pyplot as plt

"""
acc_list1 = np.array([0.1, 0.2, 0.3])
acc_list2 = np.array([0.2, 0.15, 0.1])
acc_list3 = np.array([0.3, 0.4, 0.4])
x = len(acc_list1)

# 그래프 그리기
x = np.arange(len(acc_list1))
plt.plot(x, acc_list1, marker='o', label="baseline")
plt.plot(x, acc_list2, marker="D", label= "reverse")
plt.plot(x, acc_list3, marker="v", label= "reverse + peeky")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='upper left')
plt.show()
"""

a  = np.array([0.8, 0.1, 0.03, 0.05, 0.02])
ar = a.reshape(5,1).repeat(4,axis=1)

print(a)
print(ar)