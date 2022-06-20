import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import warnings
warnings.filterwarnings("ignore")


KERNEL = {
    'laplacian': np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]]),

    'rotate': {
                'horizontal': np.array([[-1, -1, -1],
                                        [2, 2, 2],
                                        [-1, -1, -1]]),
                '45degree' : np.array([[2, -1, -1],
                                       [-1, 2, -1],
                                       [-1, -1, 2]]),
                'vertical' : np.array([[-1, 2, -1],
                                       [-1, 2, -1],
                                       [-1, 2, -1]]),
                '315degree' : np.array([[-1, -1, 2],
                                        [-1, 2, -1],
                                        [2, -1, -1]])
    },

    #Kernel filter using in edge detection

    'grad' : {
        'left' : np.array([[-1, 0],
                           [1, 0]]),
        'right' : np.array([[-1, 1],
                            [0, 0]])
    },

    'roberts' : {
        'left': np.array([[-1, 0],
                          [0, 1]]),
        'right': np.array([[0, -1],
                           [1, 0]])
    },


    'prewitt' : {
        'left' : np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]]),
        'right' : np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
    },

    'sobel' : {
        'left' : np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]]),
        'right' : np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    },


    'kirish': {
        'N' : np.array([[-3, -3, 5],
                        [-3, 0, 5],
                        [-3, -3, 5]]),
        'NW' : np.array([[-3, 5, 5],
                         [-3, 0, 5],
                         [-3, -3, -3]]),
        'W' : np.array([[5, 5, 5],
                        [-3, 0, -3],
                        [-3, -3, -3]]),
        'SW' : np.array([[5, 5, -3],
                         [5, 0, -3],
                         [-3, -3, -3]]),
        'S' : np.array([[5, -3, -3],
                        [5, 0, -3],
                        [5, -3, -3]]),
        'SE' : np.array([[-3, -3, -3],
                         [5, 0, -3],
                         [5, 5, -3]]),
        'E' : np.array([[-3, -3, -3],
                        [-3, 0, -3],
                        [5, 5, 5]]),
        'NE' : np.array([[-3, -3, -3],
                         [-3, 0, 5],
                         [-3, 5, 5]])
    }
}


def display_RGB(image: np.array) -> None:
    """ This function is for displaying rgb image
        Complex np.array will also be converted into real np.array
    Args:
        image (np.array)
    """
    if True in np.iscomplex(image):
        image = np.log(np.abs(image)+1)
        image = image*255//np.max(image)
        image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)


def display_GRAY(image: np.array) -> None:
    """ This function is for displaying gray image
        If image shape = 3: will be converted to gray before plot 
        Else: plot directly
        Complex np.array will also be converted into real np.array

    Args:
        image (np.array)
    """
    if True in np.iscomplex(image):
        image = np.log(np.abs(image)+1)
        image = image*255//np.max(image)
        image = image.astype('uint8')
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap='gray')


def multiplot(plot_name: str, images: dict) -> None:
    """ 
    This function is to plot multiple images in one plot
    Args:
        plot_name (str): name of plot
        images: dict of images to plot in format {"image_title": images}

    Cases:
        shape of 2: plot gray image automatically even when "rgb" in title
        shape of 3: plot rgb image automatically if no "gray" in title
        'gray' in image title: plot gray image
        'rgb' in image title: plot rgb image
    """

    #! align images into grid
    no_images = len(images)
    if no_images > 3:
        nrow = math.ceil(no_images/3)
        ncol = 3
    else:
        nrow = 1
        ncol = no_images
    figsize = (3*ncol, 3*nrow+1)
    plt.figure(figsize=figsize)
    plt.suptitle(f'{plot_name}')

    #! processing and plotting image in right color channel
    for title, image in images.items():
        plt.subplot(nrow, ncol, list(images.keys()).index(title) + 1)
        plt.title(f"{title}")
        plt.axis('off')

        if len(image.shape) == 2:
            display_GRAY(image)
        elif len(image.shape) == 3:
            if 'rgb' in title.lower():
                display_RGB(image)
            elif 'gray' in title.lower():
                display_GRAY(image)
            else:
                display_RGB(image)
    plt.tight_layout()
    plt.show()


class convolution2d():
    """
        This class contains convolution method in 2D-array
        Args:
            img (np.array) : 2D-array
            kernel (str or np.array): a kernel filter using in convolution (etc: Laplacian, Roberts,.. or yours)

        Method:
            convolution
    """

    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape


    """
        This function use of convolve 2D-array with a kernel filter
    """

    def convolution(self, kernel, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = self.img.shape[0]
        yImgShape = self.img.shape[1]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((self.img.shape[0] + padding * 2, self.img.shape[1] + padding * 2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = self.img
        else:
            imagePadded = self.img

        # Iterate through image
        for y in range(self.img.shape[1]):
            # Exit Convolution
            if y > self.img.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(self.img.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > self.img.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output


class smooth_threshold():
    def __init__(self, img):
        self.img = img

    def smooth(self, kernel_size):
        h, w = self.img.shape

        h = h - kernel_size + 1
        w = w - kernel_size + 1

        out_img = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                out_img[i][j] = np.mean(self.img[i:i+kernel_size,j:j+kernel_size])

        return out_img


    """
        This function return a binary image get by threshold the input image
    """
    def threshold(self, arr, thresh=0.33):
        h, w = arr.shape

        out_img = np.zeros_like(arr)

        T = thresh * np.max(arr)

        for i in range(h):
            for j in range(w):
                out_img[i][j] = 0 if arr[i][j] < T else 255

        return out_img