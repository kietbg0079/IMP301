import numpy as np





KERNEL = {
    'laplacian' : np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]]),
    'horizontal': np.array([[-1, -1, -1],
                            [2, 2, 2],
                            [-1, -1, -1]]),
    '45degree': np.array([[2, -1, -1],
                         [-1, 2, -1],
                         [-1, -1, 2]])

}


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
            raise ValueError(f'padding size must be 2 or 4 instead of {pad_size}')

        bounding_len = pad_size // 2
        img_pad = np.zeros((h + pad_size, w + 2))
        img_pad[bounding_len: h + bounding_len, bounding_len: w + bounding_len] = self.img

        if self.padding=='nn':
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
                new_image[i][j] = np.sum(img_pad[i:i + self.m, j:j + self.m] * self.kernel)

        return new_image

