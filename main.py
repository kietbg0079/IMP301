from src.filter import edge_based
from src.utils import *

import cv2
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":

    img = cv2.imread('data/10.2/img4.png', 0)

    st = smooth_threshold(img)

    smooth_img = st.smooth(5, "median")
    edge = edge_based(smooth_img)

    edge_img = edge.edge_detection('sobel')

    thresh_img = st.threshold(edge_img)

    multiplot("sboel_kernel_with_smooth", images={"origin_rgb": img, "smooth": smooth_img,"edge_img" : edge_img, "thresh_img" : thresh_img})
