import numpy as np


def fill_conv(module, layer):
    module.kernel.value[...] = np.array(layer.blobs[0].data).reshape(module.kernel.value.shape)
    module.bias.value[...] = np.array(layer.blobs[1].data).reshape(module.bias.value.shape)


def fill_relu(module, layer):
    module.coef_.value[...] = np.array(layer.blobs[0].data).reshape(module.coef_.value.shape)


def fill_linear(module, layer):
    module.weight.value[...] = np.array(layer.blobs[0].data).reshape(module.weight.value.shape)
    module.bias.value[...] = np.array(layer.blobs[1].data).reshape(module.bias.value.shape)


def nms(sort_bbox, threshold):
    sort_bbox = sorted(sort_bbox, key=lambda x: x.score, reverse=True)
    is_del = [False] * len(sort_bbox)
    output = []
    for i in range(len(sort_bbox)):
        if is_del[i]:  # 先判断是否已经删除
            continue
        for j in range(i, len(sort_bbox)):
            if is_del[j]:
                continue
            iou = sort_bbox[i] ^ sort_bbox[j]
            if iou > threshold:
                is_del[j] = True
        output.append(sort_bbox[i])
    return output


class Parameter:
    def __init__(self, value):
        self.value = value
        self.delta = np.zeros_like(self.value)


class BBox:
    def __init__(self, x, y, r, b, score=0):
        self.x, self.y, self.r, self.b, self.score = x, y, r, b, score
        self.landmarks = []

    def __xor__(self, other):
        '''
        计算box和other的IoU
        '''
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)

    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross
        return union

    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        xmax = min(self.r, other.r)
        ymax = min(self.b, other.b)
        xmin = max(self.x, other.x)
        ymin = max(self.y, other.y)
        cross_box = BBox(xmin, ymin, xmax, ymax)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0

        return cross_box.area

    def locations(self):
        return self.x, self.y, self.r, self.b

    @property
    def center(self):
        return (self.x + self.r) / 2, (self.y + self.b) / 2

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    def __repr__(self):
        return f"{{{self.x:.2f}, {self.y:.2f}, {self.r:.2f}, {self.b:.2f}, {self.score:.2f}}}"
