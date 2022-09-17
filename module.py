import numpy as np
from mtcnn import caffe_pb2 as pb


class Module:
    def __init__(self, name):
        self.name = name

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    def __init__(self, *args, **kwargs):
        super(ModuleList, self).__init__('ModuleList')
        self.module_list = args  # 元祖形式

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

    def modules(self):
        return self.module_list





