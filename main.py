from src.filter import *
from src.utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)

if __name__ == "__main__":

    img = cv2.imread('data/10.2/img4.png', 0)

    marr = Marr_hildreth(img)

    res = marr.DoG(7, 2, 6, 1)


    multiplot("canny_edge", {"origin": img, "canny edge" : res})

