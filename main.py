from src.filter import *
from src.utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)

if __name__ == "__main__":

    img = cv2.imread('data/10.2/img5.png', 0)

    can = canny(img)

    mag, arctan, K, strong, res = can.canny(0.05, 0.15)


    multiplot("canny_edge", {"origin": img, "magnitude": mag, "angle": arctan, "nonmax_supression": K, "result": res})

