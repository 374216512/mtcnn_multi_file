import cv2
import numpy as np
from module import Module, ModuleList
from conv import Conv2d
from pooling import MaxPooling2d
from activation import Softmax, PRelu
from mtcnn import caffe_pb2 as pb
from utils import fill_relu, fill_conv, fill_linear
from linear import Linear


class CaffeLoader(Module):
    def __init__(self, path):
        super(CaffeLoader, self).__init__('CaffeLoader')
        self.path = path

    def load(self, *modules):
        # 先加载caffemodel
        net = pb.NetParameter()
        with open(self.path, 'rb') as f:
            net.ParseFromString(f.read())

        # 让caffemodel的name和layer做一个映射
        layer_map = {layer.name: layer for layer in net.layer}

        # 对module进行赋值
        for module_name, module in modules:
            if isinstance(module, Conv2d):
                fill_conv(module, layer_map[module_name])
            if isinstance(module, PRelu):
                fill_relu(module, layer_map[module_name])
            if isinstance(module, Linear):
                fill_linear(module, Conv2d)


class PNet(Module):
    def __init__(self, path):
        super(PNet, self).__init__('PNet')
        self.backbone = ModuleList(
            Conv2d(3, 10, 3),
            PRelu(10),
            MaxPooling2d(),
            Conv2d(10, 16, 3),
            PRelu(16),
            Conv2d(16, 32, 3),
            PRelu(32)
        )
        self.confidence = Conv2d(32, 2, 1)
        self.softmax = Softmax()
        self.bbox = Conv2d(32, 4, 1)

        self.caffe_loader = CaffeLoader(path)
        self.layer_name = ['conv1', 'PReLU1', 'pool1', 'conv2', 'PReLU2', 'conv3', 'PReLU3']
        self.caffe_loader.load(
            *list(zip(self.layer_name, self.backbone.modules())),
            *list(zip(['conv4-1', 'conv4-2'], [self.confidence, self.bbox]))
        )

    def forward(self, x):
        x = self.backbone(x)
        confidence = self.softmax(self.confidence(x))
        bbox = self.bbox(x)
        return confidence, bbox


class RNet(Module):
    def __init__(self, path):
        super(RNet, self).__init__('RNet')
        self.backbone = ModuleList(
            Conv2d(1, 28, 3),
            PRelu(28),
            MaxPooling2d(),
            Conv2d(28, 48, 3),
            PRelu(48),
            MaxPooling2d(),
            Conv2d(48, 64, 2),
            PRelu(64),
            Linear(64, 128),
            PRelu(128)
        )
        self.confidence = Linear(128, 2)
        self.bbox = Linear(128, 4)
