import cv2 as cv
import numpy as np


# 3x3==》》4x4
def img_resize(src, new_size):
    dst_w, dst_h = new_size  # src_h
    src_h, src_w = src.shape[:2]  # src_w
    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    scale_x = float(src_w) / dst_w  # x缩放比例
    scale_y = float(src_h) / dst_h  # y缩放比例

    # 遍历目标图像，插值
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for n in range(3):  # 对channel循环
        for dst_y in range(dst_h):  # 对height循环
            for dst_x in range(dst_w):  # 对width循环
                # 目标在源上的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 计算在源图上四个近邻点的位置
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_1 = min(src_y_0 + 1, src_h - 1)

                # 双线性插值
                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst


if __name__ == '__main__':
    img_in = cv.imread('../datas/lenna.png')
    img_out = img_resize(img_in, (800, 800))
    cv.imshow('src_image', img_in)
    cv.imshow('dst_image', img_out)
    cv.waitKey(0)
