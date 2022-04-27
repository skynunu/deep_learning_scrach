import os
import sys
sys.path.append(os.pardir)  
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from optimizer import *


graph_draw_num = 20
col_num =np.array([2.3,-2.1,4.0])
row_num = np.ceil(col_num)
print(row_num)