import cv2
import numpy as np

from module import Module
from utils import Parameter


class Conv2d(Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride=1, padding=0):
        # in_feature: C
        # out_feature: filter_num
        super(Conv2d, self).__init__('Conv2d')
        self.ob = None
        self.oc = None
        self.column = None
        self.oh = None
        self.ow = None
        self.in_shape = None
        self.output = None
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = Parameter(
            np.zeros((self.out_feature, self.in_feature, self.kernel_size, self.kernel_size))
        )
        self.bias = Parameter(
            np.zeros(self.out_feature)
        )

    def forward(self, x):
        # 1、保存输入图片的维度
        self.in_shape = x.shape
        ib, ic, ih, iw = self.in_shape
        # 2、计算输出图片的维度
        self.oh = (ih + self.padding * 2 - self.kernel_size) // self.stride + 1
        self.ow = (iw + self.padding * 2 - self.kernel_size) // self.stride + 1
        self.ob = ib
        self.oc = self.out_feature
        # 3、计算im2col转换之后的矩阵维度
        col_w = self.oh * self.ow
        col_h = self.kernel_size * self.kernel_size * self.in_feature
        # 4、初始化数据
        self.column = np.zeros((ib, col_h, col_w))  # im2col
        self.output = np.zeros((self.ob, self.oc, self.oh, self.ow))  # 输出矩阵
        kernel_col = self.kernel.value.reshape(self.out_feature, -1)

        for b in range(ib):
            # 一张图填充column的一个通道
            for c in range(ic):
                for oy in range(self.oh):
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                                # 计算column中的下标
                                column_y = c * self.kernel_size * self.kernel_size + ky * self.kernel_size + kx
                                column_x = ox + oy * self.ow
                                # 计算input_image中的下标
                                iy = oy * self.stride + ky - self.padding
                                ix = ox * self.stride + kx - self.padding
                                # 如果没有越界，则做出im2col的映射
                                if 0 <= ix <= iw and 0 <= iy <= ih:
                                    self.column[b, column_y, column_x] = x[b, c, iy, ix]
            # 转换完一张图之后进行GEMM计算
            self.output[b] = (kernel_col @ self.column[b]).reshape(self.out_feature, self.oh, self.ow) \
                             + self.bias.value.reshape(self.out_feature, 1, 1)
        return self.output


if __name__ == '__main__':
    image = cv2.imread('../own/my.jpg').transpose(2, 0, 1)[None]
    conv = Conv2d(3, 10, 3)
    output = conv(image)
    print(output.shape)
