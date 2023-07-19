import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from layer import CannyEdge
import matplotlib

matplotlib.use("Qt5Agg")

image = cv2.imread('test_img.jpg')
image = cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB
)
image = tf.image.rgb_to_grayscale(
    tf.convert_to_tensor(
        image, dtype=tf.float32
    )
)
image = tf.reshape(
    image, (1, *image.shape[:-1], 1)
)

edges_opencv = cv2.Canny(
    np.uint8(image.numpy()[0]), 50, 100
)
edge_tf = CannyEdge(
    max_val=100, min_val=50, connection_iterations=10, sigma=0.5
)(image)

fig, ax = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(wspace=0)
ax[0].imshow(image.numpy()[0].astype(np.uint8), cmap='gray')
ax[1].imshow(edges_opencv.astype(np.uint8), cmap='gray')
ax[2].imshow(edge_tf.numpy()[0].astype(np.uint8), cmap='gray')