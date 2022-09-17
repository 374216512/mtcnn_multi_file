import cv2

from module import Module
import numpy as np


class MaxPooling2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        # 默认池化核大小为2，步长为2
        # 这样的池化核会使输入图片的宽高减半，面积减小为原来的1/4
        super(MaxPooling2d, self).__init__('MaxPooling2d')
        self.output = None
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # 1、计算输入图片的维度
        ib, ic, ih, iw = x.shape
        # 2、计算输出图片的维度
        ob = ib
        oc = ic
        oh = int(np.ceil((ih - self.kernel_size) / self.stride) + 1)
        ow = int(np.ceil((iw - self.kernel_size) / self.stride) + 1)
        # 3、初始化输出数组
        self.output = np.zeros((ob, oc, oh, ow))
        for b in range(ib):
            for c in range(ic):
                for oy in range(oh):
                    for ox in range(ow):
                        # 假设每次滑动窗口的最左上角的值为最大值，并填入output矩阵
                        ix = ox * self.stride
                        iy = oy * self.stride
                        self.output[b, c, oy, ox] = x[b, c, iy, ix]
                        # 遍历滑动窗口的每一个值，找出最大值，填入到output矩阵
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                                ix = ox * self.stride + kx
                                iy = oy * self.stride + ky
                                if ix < iw and iy < ih:
                                    self.output[b, c, oy, ox] = max(x[b, c, iy, ix], self.output[b, c, oy, ox])
        return self.output


if __name__ == '__main__':
    image = cv2.imread('../own/my.jpg').transpose(2, 0, 1)[None]
    pool = MaxPooling2d()
    output = pool(image)
    print(output.shape)
