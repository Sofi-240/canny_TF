import os
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from layer import CannyEdge
import matplotlib

matplotlib.use("Qt5Agg")


def load_test():
    imgs = []
    for file in os.listdir('test_data'):
        image = cv2.imread(
            os.path.join(
                'test_data', file
            )
        )
        image_shape = image.shape
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )
        image = tf.image.rgb_to_grayscale(
            tf.convert_to_tensor(
                image, dtype=tf.float32
            )
        )
        image = tf.reshape(
            image, (1, *image_shape[:-1], 1)
        )
        imgs.append(image)
    return imgs


def show(*imgs):
    fig, _ = plt.subplots(1, len(imgs), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0)
    for i, im in enumerate(imgs):
        if tf.is_tensor(im):
            fig.axes[i].imshow(im.numpy()[0].astype(np.uint8), cmap='gray')
        else:
            fig.axes[i].imshow(im.astype(np.uint8), cmap='gray')


images = load_test()

image = images[1]
edges_opencv = cv2.Canny(np.uint8(image.numpy()[0]), 50, 100)
X = CannyEdge(max_val=100, min_val=50)
X.build(input_shape=image.shape)

edge_image, edge_week, edge_sure = X(image)

show(image, edge_image, edges_opencv)
show(edge_sure, edge_week)

# kernel = tf.constant(
#     [
#         [1, 1, 1, 1, 1],
#         [1, 1, 0, 1, 1],
#         [1, 0, 0, 0, 1],
#         [1, 1, 0, 1, 1],
#         [1, 1, 1, 1, 1]
#     ],
#     dtype=tf.float32
# )
# kernel = tf.reshape(
#     kernel, shape=(5, 5, 1)
# )
# kernel = tf.constant(
#     [
#         [1, 1, 1],
#         [1, 0, 1],
#         [1, 1, 1]
#     ],
#     dtype=tf.float32
# )
# kernel = tf.reshape(
#     kernel, shape=(3, 3, 1)
# )
# connected = tf.nn.erosion2d(
#     edge_sure,
#     kernel,
#     (1, 1, 1, 1),
#     'SAME',
#     'NHWC',
#     (1, 1, 1, 1)
# ) + 1
#
# edge = tf.where(
#     tf.math.logical_or(
#         tf.math.equal(
#             connected, 255.0
#         ),
#         tf.math.equal(
#             edge_sure, 255.0
#         )
#     ), 255.5, 0.0
# )
# show(connected, edge, edges_opencv)

