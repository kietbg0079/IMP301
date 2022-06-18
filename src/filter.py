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


class region_growing:
    def __init__(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtClor(img, cv2.COLOR_BGR2GRAY)
        self.img = img
        self.image_shape = img.shape

    def get8neighbor(self, x, y):
        """This function re turn list of neighbors of input point

        Args:
            x (int): x-cordinate value of point
            y (int): y-cordinate value of point
            image_shape (tuple): (height, width) of processing image

        Returns:
            list: list of neighbors of input point
        """
        neighbors = []
        maxx = self.image_shape[1]-1
        maxy = self.image_shape[0]-1

        # top left
        outx = min(max(x-1, 0), maxx)
        outy = min(max(y-1, 0), maxy)
        neighbors.append((outx, outy))

        # top center
        outx = x
        outy = min(max(y-1, 0), maxy)
        neighbors.append((outx, outy))

        # top right
        outx = min(max(x+1, 0), maxx)
        outy = min(max(y-1, 0), maxy)
        neighbors.append((outx, outy))

        # left
        outx = min(max(x-1, 0), maxx)
        outy = y
        neighbors.append((outx, outy))

        # right
        outx = min(max(x+1, 0), maxx)
        outy = y
        neighbors.append((outx, outy))

        # bottom left
        outx = min(max(x-1, 0), maxx)
        outy = min(max(y+1, 0), maxy)
        neighbors.append((outx, outy))

        # bottom center
        outx = x
        outy = min(max(y+1, 0), maxy)
        neighbors.append((outx, outy))

        # bottom right
        outx = min(max(x+1, 0), maxx)
        outy = min(max(y+1, 0), maxy)
        neighbors.append((outx, outy))

        return neighbors

    def getSeeds(self, percentile=99):
        """This function finds seeds for region-growing algorithm

        Args:
            img (np array): processing image
            percentile (int, optional): nth percentile value. Defaults to 99.

        Returns:
            centroid (list): list of seeds for region-growing
            percentile_value (int): value of percentile intensity
        """
        seeds = []

        percentile_value = np.percentile(self.img, percentile)
        _, thresh = cv2.threshold(
            self.img, percentile_value, 255, cv2.THRESH_BINARY)
        analysis = cv2.connectedComponentsWithStats(thresh,
                                                    8,
                                                    cv2.CV_32S)
        (_, _, _, centroid) = analysis
        print(centroid.shape)
        return np.round(centroid).astype(int), percentile_value

    def region_growing(self, seeds=None, seed_value=255, T1=68, T2=126):
        if seeds is None:
            seeds, seed_value = self.getSeeds(self.img, 99)

        outimg = np.zeros_like(self.img)
        diff = np.abs(self.img - seed_value)
        for seed in seeds:
            seed_points = []
            seed_points.append((seed[1], seed[0]))
            processed = []
            while(len(seed_points) > 0):
                pix = seed_points[0]
                outimg[pix[0], pix[1]] = seed_value
                for coord in self.get8neighbor(pix[0], pix[1]):
                    if diff[coord[0], coord[1]] <= T1 or diff[coord[0], coord[1]] <= T2:
                        outimg[coord[0], coord[1]] = 255
                        if not coord in processed:
                            seed_points.append(coord)
                    processed.append(coord)
                seed_points.pop(0)
        return outimg
