import matplotlib.pyplot as plt
import math
import cv2

def multiplot(plot_name: str,images: dict,color_channel: str = None) -> None:
    """ 
    This function is to plot multiple images in one plot
    Args:
        plot_name (str): name of plot
        images: images to plot in format {"image_title": images: np.array}
    """
    no_images = len(images)
    if no_images > 3:
        nrow = math.ceil(no_images/3)
        ncol = 3
    else:
        nrow = 1
        ncol = no_images
    figsize = (3*ncol + 2, 3*nrow + 1)
    plt.figure(figsize=figsize)
    plt.suptitle(f'{plot_name}')

    for title, image in images.items():
        plt.subplot(nrow,ncol,list(images.keys()).index(title) + 1)
        plt.title(f"{title}")
        assert color_channel in ['rgb','gray', None]
        if not color_channel:
            if 'rgb' in title and len(image.shape) == 3:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                plt.imshow(image)
            elif 'gray' in title:
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                plt.imshow(image,cmap = 'gray')
        elif color_channel == 'rgb':
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            plt.imshow(image)
        elif color_channel == 'gray':
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.imshow(image,cmap = 'gray')
    
    plt.show()