from src.filter import *
from src.utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    img = cv2.imread('data/10.2/test.jpg')

    morp = morphological(img)

    out_img = morp.morphol()

    multiplot("Marr and Hildreth", {"origin" : img, "morphological" : out_img})


