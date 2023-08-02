import cv2
from matplotlib import pyplot as plt
from canny import *
import matplotlib
import tensorflow as tf
import numpy as np

matplotlib.use("Qt5Agg")


def show_images(images, subplot_x=1, subplot_y=1):
    def show(image, ax):
        if tf.is_tensor(image):
            image = image.numpy()
        if np.max(image) > 1:
            image = image.astype(np.uint8)
        if len(image.shape) == 2 or image.shape[-1] == 1:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)

    fig, _ = plt.subplots(subplot_y, subplot_x, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(subplot_x * subplot_y):
        show(images[i], fig.axes[i])


img = cv2.imread('V2_2.jpg', 0)

# tf.config.run_functions_eagerly(True)

edges_con = canny_edge(img,
                       sigma=0.8,
                       kernel_size=5,
                       min_val=50,
                       max_val=100,
                       hysteresis_tracking_alg='connection',
                       tracking_con=15,
                       tracking_iterations=None)

edges_dil = canny_edge(img,
                       sigma=0.8,
                       kernel_size=5,
                       min_val=50,
                       max_val=100,
                       hysteresis_tracking_alg='dilation',
                       tracking_con=5,
                       tracking_iterations=20)

edges_opencv = cv2.Canny(img, 50, 100)
show_images([edges_opencv, edges_con[0, ...], edges_dil[0, ...]], subplot_x=3, subplot_y=1)
