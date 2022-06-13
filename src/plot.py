import matplotlib.pyplot as plt
import math


def multiplot(plot_name: str,images: dict,color_channel: str) -> None:
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

    for title, image in images:
        plt.subplot(nrow,ncol,list(images.keys()).index(title) + 1)
        plt.title(f"{title}")
        plt.imshow(image)
    