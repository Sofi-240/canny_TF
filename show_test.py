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


img = cv2.imread('test_img.jpg', 0)

edge_tf = canny_edge(img, max_val=100, min_val=50, connection_iterations=20, kernel_size=5, sigma=0.8)

edges_opencv = cv2.Canny(img, 50, 100)
show_images([edges_opencv, edge_tf[0, ...]], subplot_x=2, subplot_y=1)

