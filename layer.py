import tensorflow as tf
import numpy as np


def gaussian_kernel(channels, kernel_size, sigma):
    ax = tf.range(
        -kernel_size // 2 + 1.0, kernel_size // 2 + 1.0
    )
    xx, yy = tf.meshgrid(ax, ax)
    k = tf.exp(
        -(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)
    )
    k = k / tf.reduce_sum(k)
    k = tf.tile(
        k[..., tf.newaxis, tf.newaxis], [1, 1, channels, 1]
    )
    return k


class CannyEdge(tf.keras.layers.Layer):
    def __init__(self, sigma=0.05, kernel_size=3, min_val=50, max_val=100, **kwargs):
        super(CannyEdge, self).__init__(**kwargs)
        self._input_shape = None
        self.karnel = None
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.min_val = min_val
        self.max_val = max_val

    def build(self, input_shape):
        self._input_shape = input_shape
        self.karnel = gaussian_kernel(
            channels=self._input_shape[-1],
            kernel_size=self.kernel_size,
            sigma=self.sigma
        )

    def call(self, images, *args, **kwargs):
        if self._input_shape is None:
            self.build(images.shape)

        images = tf.pad(
            images,
            tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
            mode='REFLECT'
        )
        images_blur = tf.nn.convolution(
            images, self.karnel, padding='VALID'
        )
        gx, gy, gxy = self._sobel_edge(images_blur)

        theta = tf.atan2(gx, gy)
        theta = ((theta * 180 / np.pi) + 90) % 180

        ang_cond = [
            (tf.math.logical_or, (157.5, 22.5)),
            (tf.math.logical_and, (22.5, 67.5)),
            (tf.math.logical_and, (67.5, 112.5)),
            (tf.math.logical_and, (112.5, 157.5))
        ]
        filters = [
            [
                [-np.inf, -np.inf, -np.inf],
                [0.0, 0.0, 0.0],
                [-np.inf, -np.inf, -np.inf]
            ],
            [
                [-np.inf, -np.inf, 0.0],
                [-np.inf, 0.0, -np.inf],
                [0.0, -np.inf, -np.inf]
            ],
            [
                [-np.inf, 0.0, -np.inf],
                [-np.inf, 0.0, -np.inf],
                [-np.inf, 0.0, -np.inf]
            ],
            [
                [0.0, -np.inf, -np.inf],
                [-np.inf, 0.0, -np.inf],
                [-np.inf, -np.inf, 0.0]
            ],
        ]
        edge_before_thresh = None

        for ang, kernel in zip(ang_cond, filters):
            temp_ang = ang[0](
                tf.math.greater_equal(
                    theta, ang[1][0]
                ),
                tf.math.less(
                    theta, ang[1][1]
                )
            )
            ang_image = tf.where(
                temp_ang, gxy, 0.0
            )
            ang_image_pad = tf.keras.layers.ZeroPadding2D(
                padding=1
            )(ang_image)

            kernel = tf.reshape(
                tf.constant(
                    kernel, tf.float32
                ), [3, 3, 1]
            )
            max_pool_ang = tf.nn.dilation2d(
                ang_image_pad,
                kernel, (1, 1, 1, 1),
                'VALID',
                'NHWC',
                (1, 1, 1, 1)
            )

            is_local_max = tf.math.equal(
                max_pool_ang, ang_image
            )
            is_max = tf.where(
                is_local_max, ang_image, 0.0
            )
            if edge_before_thresh is None:
                edge_before_thresh = is_max
            else:
                edge_before_thresh = edge_before_thresh + is_max
        edge_sure = tf.where(
            tf.math.greater_equal(
                edge_before_thresh, self.max_val
            ), 255.0, 0.0
        )

        edge_week = tf.where(
            tf.math.logical_and(
                tf.math.greater_equal(
                    edge_before_thresh, self.min_val
                ),
                tf.math.less(
                    edge_before_thresh, self.max_val
                )
            ), 255.0, 0.0
        )
        # connection between the edges heer
        edge_image = tf.where(
            tf.math.logical_or(
                tf.math.equal(
                    edge_week, 255.0
                ),
                tf.math.equal(
                    edge_sure, 255.0
                )
            ), 255.5, 0.0
        )
        return edge_image, edge_week, edge_sure

    def _sobel_edge(self, images):
        channels = self._input_shape[-1]
        images = tf.pad(
            images,
            tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
            mode='REFLECT'
        )
        h = tf.constant(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], tf.float32
        )
        v = tf.constant(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], tf.float32
        )
        h = tf.reshape(
            h, [3, 3, channels, 1]
        )
        v = tf.reshape(
            v, [3, 3, channels, 1]
        )
        gx = tf.nn.convolution(
            images, h, padding='VALID'
        )
        gy = tf.nn.convolution(
            images, v, padding='VALID'
        )
        gxy = tf.sqrt(
            tf.square(gx) + tf.square(gy)
        )
        return gx, gy, gxy
