import cv2
from .utils import *
import numpy as np



class edge_based(convolution2d):
    """"
        This class contain edge_based approach in image segmentation base on discontinuity property
    """
    def __init__(self, img):
        super().__init__(img)
        self.img = img


    """"
        This function can be use in either noise or isolated point detection
        
        Args:
            T_threshold : threshold value for filtering convolutional image
            kernel (str or np.array): a kernel filter
    """
    def isolated_point(self, T_threshold):

        img_filter = self.convolution(KERNEL['laplacian'])

        a, b = img_filter.shape
        T = T_threshold * np.max(img_filter.flatten())


        bin_img = np.zeros_like(img_filter)
        for i in range(a):
            for j in range(b):
                bin_img[i][j] = 0 if img_filter[i][j] < T else 1

        return bin_img


    """
        This function use of detect either edge or line
        
        kernel (str) : filter kernel (prewitt, roberts, sobel)
    """

    def kirish(self):
        lst_img = []

        for key, value in KERNEL['kirish'].items():
            lst_img.append(self.convolution(value))

        a, b = lst_img[0].shape

        out_img = np.zeros_like(lst_img[0])
        for i in range(a):
            for j in range(b):
                out_img[i][j] = np.max([arr[i][j] for arr in lst_img])

        return out_img

    def edge_detection(self, kernel):

        if len(KERNEL[kernel]) == 2:
            img_left = self.convolution(KERNEL[kernel]['left'])
            img_right = self.convolution(KERNEL[kernel]['right'])

        elif kernel == "kirish":
            return self.kirish()

        else:
            raise ValueError("This filter can not use in edge detection!")

        a, b = img_left.shape

        edge_img = np.zeros_like(img_left)
        for i in range(a):
            for j in range(b):
                edge_img[i][j] = np.abs(img_left[i][j]) + np.abs(img_right[i][j])

        return edge_img