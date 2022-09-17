import numpy as np
from module import Module
from utils import Parameter


class Linear(Module):
    def __init__(self, in_feature, out_feature):
        super(Linear, self).__init__("Linear")
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(np.zeros((in_feature, out_feature)))
        self.bias = Parameter(np.zeros(out_feature))

    def forward(self, x):
        x = x @ self.weight.value + self.bias.value
        return x
