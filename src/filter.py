import cv2
from .utils import *
import numpy as np
import scipy.ndimage


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
                edge_img[i][j] = np.abs(
                    img_left[i][j]) + np.abs(img_right[i][j])

        return edge_img


class marr_hildreth():
    def __init__(self, img):
        self.img = img

    """
        A Gaussian function (kernel) received from Gaussian distribution
        
        Args:
            std (float): standard deviation of gaussian distribution
            
        Return:
            np.array : with size 6*std x 6*std   
    """

    def gaus_dis(self, std=4):

        #kernel_size = 6*std if (6*std) % 2 == 1 else 6*std+1
        kernel_size = 29
        gaus_kernel = np.zeros((kernel_size, kernel_size))

        for i in range(kernel_size):
            for j in range(kernel_size):
                gaus_kernel[i][j] = np.exp(-(i**2 + j**2) / (2*std**2))

        return gaus_kernel

    """
        This function use of calculating the different of Gaussian with 
        std_a and std_b use for calculating DoG image scale x, std_c and 
        std_d are inputs of DoG image scale y
        
        Args:
            std_a, std_b, std_c, std_d: Standard deviation of Gaussian function
            
        Returns:
            binary 2D-array : where 1 if its pixel is a potential keypoint 0 for
            otherwise
    
    """

    def DoG(self, std_a, std_b, std_c, std_d):
        img_a = convolution2d(self.img).convolution(self.gaus_dis(std_a))
        img_b = convolution2d(self.img).convolution(self.gaus_dis(std_b))

        dog_img_x = img_a - img_b

        img_c = convolution2d(self.img).convolution(self.gaus_dis(std_c))
        img_d = convolution2d(self.img).convolution(self.gaus_dis(std_d))

        dog_img_y = img_c - img_d

        h, w = dog_img_y.shape

        out_img = np.zeros((h, w))
        for i in range(1, h-1):
            for j in range(1, w-1):
                dog_x_max = np.max(dog_img_x[i-1:i+1, j-1:j+1])
                dog_y_max = np.max(dog_img_y[i-1:i+1, j-1:j+1])

                if dog_img_x[i][j] == max(dog_img_x[i][j], dog_x_max, dog_y_max):
                    out_img[i][j] = 1
                else:
                    out_img[i][j] = 0

        return out_img


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


class Split_Merge_Segmented:
    def __init__(self, img):
        self.img = img

    # split
    def Split_Judge(self, h0, w0, h, w):
        """This function will consider if region is splittable

        Args:
            h0 (int): starting point of region to check (y-coordinate)
            w0 (int): starting point of region to check (x-coordinate)
            h (int): height of region to check
            w (int): width of region to check

        Returns:
            bool:   True if region can still be splitted
                    False if region can not be splitted anymore
        """
        area = self.img[h0: h0 + h, w0: w0 + w]
        mean = np.mean(area)
        std = np.std(area, ddof=1)

        total_points = 0
        operated_points = 0

        for row in range(area.shape[0]):
            for col in range(area.shape[1]):
                if (area[row][col] - mean) < 2 * std:
                    operated_points += 1
                total_points += 1

        if operated_points / total_points >= 0.95:
            return True
        else:
            return False

    def Merge(self, h0, w0, h, w):
        """This function merges regions together

        Args:
            h0 (int): starting point of region to merge (y-coordinate)
            w0 (int): starting point of region to merge (x-coordinate)
            h (int): height of region to merge
            w (int): width of region to merge
        """
        img = self.img
        for row in range(h0, h0 + h):
            for col in range(w0, w0 + w):
                if img[row, col] > 100 and img[row, col] < 200:
                    img[row, col] = 0
                else:
                    img[row, col] = 255

    def Recursion(self, h0, w0, h, w):
        """This function continually split region until stopping condition are met

        Args:
            h0 (int): starting point of region to consider (y-coordinate)
            w0 (int): starting point of region to consider (x-coordinate)
            h (int): height of region to consider
            w (int): width of region to consider
        """
        if not self.Split_Judge(h0, w0, h, w) and min(h, w) > 5:
            # Recursion continues to determine whether it can continue to split
            # Top left square
            self.Recursion(h0, w0, int(h0 / 2), int(w0 / 2))
            # Upper right square
            self.Recursion(h0, w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
            # Lower left square
            self.Recursion(h0 + int(h0 / 2), w0, int(h0 / 2), int(w0 / 2))
            # Lower right square
            self.Recursion(h0 + int(h0 / 2), w0 + int(w0 / 2),
                           int(h0 / 2), int(w0 / 2))
        else:
            # Merge
            self.Merge(h0, w0, h, w)

    def segment(self):
        """This function will perform region splitting and region merging segmentation method and show
        """
        origin = self.img.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # hist, bins = np.histogram(img_gray, bins=256)

        segemented_img = img_gray.copy()
        self.img = segemented_img
        self.Recursion(0, 0, segemented_img.shape[0], segemented_img.shape[1])

        multiplot("Split and merge segmentation", {
                  'input image': origin, "input gray": img_gray, "segmented image": segemented_img})







class morphological():
    """
        Image segmentation using morphological wartersheed method

        Args:
            image (RGB channel): Input image
        Returns:
            image : with the object surrounded by a boundary
    """
    def __init__(self, img):
        self.img = img


    def morphol(self):

        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # background area
        sure_bg = cv2.dilate(closing, kernel, iterations=1)

        # Finding foreground area
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Applying watershed algorithm and marking the regions segmented
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(self.img, markers)

        self.img[markers == -1] = [255, 0, 0]

        return self.img