from src.filter import *
from src.utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    img = cv2.imread('data/10.2/img4.png', 0)

    mh = marr_hildreth(img)

    # gaus_fil = mh.gaus_dis()
    #
    # conv_img = convolution2d(img).convolution(gaus_fil)
    #
    # lapla = convolution2d(conv_img).convolution(KERNEL['laplacian'])
    #
    # zec = smooth_threshold(img).threshold(lapla, 0.04)

    multiplot("Marr and Hildreth", {"origin" : img, "dog" : mh.DoG(2, 4, 1, 5)})


