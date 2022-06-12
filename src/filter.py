import cv2
from .utils import *
import numpy as np



class edge_based(convolution2d):
    def __init__(self, img, kernel='laplacian'):
        super().__init__(img, KERNEL[kernel])
        self.img = img
        self.kernel = KERNEL[kernel]



    def isolated_point(self):

        img_filter = self.convolution()

        a, b = img_filter.shape
        T = 0.7 * np.max(img_filter.flatten())

        bin_img = np.zeros((self.h, self.w))
        for i in range(a):
            for j in range(b):
                bin_img[i][j] = 0 if img_filter[i][j] < T else 1

        return img_filter