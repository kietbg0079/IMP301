import cv2
from src.filter import edge_based
import matplotlib.pyplot as plt
import numpy as np






img = cv2.imread('data/10.2/img3.png', 0)

binary_img = edge_based(img).isolated_point()


plt.imshow(binary_img, cmap='gray')
plt.show()
