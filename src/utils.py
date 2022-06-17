import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import warnings
warnings.filterwarnings("ignore")


KERNEL = {
    'laplacian': np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]]),
    'horizontal': np.array([[-1, -1, -1],
                            [2, 2, 2],
                            [-1, -1, -1]]),
    '45degree': np.array([[2, -1, -1],
                         [-1, 2, -1],
                         [-1, -1, 2]])

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
    figsize = (3*ncol, 3*nrow + 2)
    plt.figure(figsize=figsize)
    plt.suptitle(f'{plot_name}')

    #! processing and plotting image in right color channel
    for title, image in images.items():
        plt.subplot(nrow, ncol, list(images.keys()).index(title) + 1)
        plt.title(f"{title}")

        if len(image.shape) == 2:
            display_GRAY(image)
        elif len(image.shape) == 3:
            if 'rgb' in title.lower():
                display_RGB(image)
            elif 'gray' in title.lower():
                display_GRAY(image)
            else:
                display_RGB(image)
    plt.show()


class convolution2d():
    def __init__(self, img, kernel, padding='zero'):
        self.img = img

        self.m, self.n = kernel.shape

        if self.m != self.n:
            raise ValueError(f'kernel size must be square inseat')
        self.kernel = kernel
        self.padding = padding

    def pad_img(self, pad_size=2):
        h, w = self.img.shape

        if (pad_size not in [2, 4]):
            raise ValueError(
                f'padding size must be 2 or 4 instead of {pad_size}')

        bounding_len = pad_size // 2
        img_pad = np.zeros((h + pad_size, w + 2))
        img_pad[bounding_len: h + bounding_len,
                bounding_len: w + bounding_len] = self.img

        if self.padding == 'nn':
            img_pad[0, :] = img_pad[1, :]
            img_pad[-1, :] = img_pad[-2, :]
            img_pad[:, 0] = img_pad[:, 1]
            img_pad[:, -1] = img_pad[:, -2]

        self.h, self.w = h, w
        self.bounding_len = bounding_len

        return img_pad

    def convolution(self):

        img_pad = self.pad_img()

        h = self.h - self.m + self.bounding_len
        w = self.w - self.m + self.bounding_len
        new_image = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                new_image[i][j] = np.sum(
                    img_pad[i:i + self.m, j:j + self.m] * self.kernel)

        return new_image
