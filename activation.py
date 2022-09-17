import numpy as np
from module import Module
from utils import Parameter


class PRelu(Module):
    def __init__(self, channel, inplace=True):
        super(PRelu, self).__init__('PRelu')
        self.channel = channel
        self.inplace = inplace
        self.coef_ = Parameter(np.zeros(self.channel))

    def forward(self, x):
        if not self.inplace:
            x = x.copy()
        for c in range(x.shape[1]):  # 一个通道一个通道的计算
            # 小于0时乘以指定数值
            negative = x[:, c] < 0
            x[:, c][negative] *= self.coef_.value[c]
        return x


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__('Softmax')

    def forward(self, x):
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp
