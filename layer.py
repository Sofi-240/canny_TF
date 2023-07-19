import tensorflow as tf
import numpy as np


class CannyEdge(tf.keras.layers.Layer):
    def __init__(self, sigma=None, kernel_size=5, min_val=50, max_val=100, connection_iterations=20, **kwargs):
        super(CannyEdge, self).__init__(**kwargs)
        self._input_shape = None
        self.gaussian_setup = dict(
            sigma=sigma, kernel_size=kernel_size
        )
        self.threshold_setup = dict(
            min_val=min_val, max_val=max_val
        )
        self.hysteresis_setup = dict(
            algorithm='dilation', connection_iterations=connection_iterations
        )
        # another algorithm option will be DFS.
        self.zero_pad = tf.keras.layers.ZeroPadding2D(
            padding=1
        )
        self.angles_names = [0, 45, 90, 135]
        self._set_up = False

    def _setup(self, input_shape):
        if input_shape[-1] != 1:
            raise ValueError(
                'The shape of the input need to be (batch_size, height, weight, 1)'
            )
        local_max_filters = [
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
        angles_call_range = [
            (tf.math.logical_or, (157.5, 22.5)),
            (tf.math.logical_and, (22.5, 67.5)),
            (tf.math.logical_and, (67.5, 112.5)),
            (tf.math.logical_and, (112.5, 157.5))
        ]

        """
        Build the gaussian kernel.
        """
        kernel_size = self.gaussian_setup.get('kernel_size')
        sigma = self.gaussian_setup.get('sigma')
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        ax = tf.range(
            -kernel_size // 2 + 1.0, kernel_size // 2 + 1.0
        )
        xx, yy = tf.meshgrid(ax, ax)
        normal = 1 / (2.0 * np.pi * (sigma ** 2))
        kernel = tf.exp(
            -((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2))
        ) * normal

        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(
            kernel[..., tf.newaxis, tf.newaxis], [1, 1, input_shape[-1], 1]
        )
        self.gaussian_kernel = kernel
        self.gaussian_pad = tf.constant(
            [[0, 0],
             [kernel_size // 2, kernel_size // 2],
             [kernel_size // 2, kernel_size // 2],
             [0, 0]]
        )

        """
        Sobel filters.
        """

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
        h = tf.tile(
            h[..., tf.newaxis, tf.newaxis], [1, 1, input_shape[-1], 1]
        )
        v = tf.tile(
            v[..., tf.newaxis, tf.newaxis], [1, 1, input_shape[-1], 1]
        )
        self.sobel_filters = tf.concat(
            (h, v), axis=-1
        )

        for kernel, call_range, angle in zip(local_max_filters, angles_call_range, self.angles_names):
            kernel = tf.reshape(
                tf.constant(
                    kernel, tf.float32
                ), [3, 3, 1]
            )
            self.__setattr__(
                f'local_max_kernel_{angle}', kernel
            )
            self.__setattr__(
                f'call_range_{angle}', call_range
            )

        self.hysteresis_kernel = tf.ones(
            shape=(5, 5, 1), dtype=tf.float32
        )
        self._set_up = True

    def build(self, input_shape):
        self._input_shape = input_shape
        self._setup(input_shape)

    def call(self, images, *args, **kwargs):
        if not self._set_up:
            self.build(images.shape)

        images = tf.pad(
            images, self.gaussian_pad, mode='REFLECT'
        )
        images_blur = tf.nn.convolution(
            images, self.gaussian_kernel, padding='VALID'
        )

        images_blur = self.zero_pad(images_blur)
        gxy = tf.nn.convolution(
            images_blur, self.sobel_filters, padding='VALID'
        )
        theta = tf.atan2(
            gxy[..., 0], gxy[..., 1]
        )
        theta = ((theta * 180 / np.pi) + 90) % 180
        gxy = tf.sqrt(
            tf.square(gxy[..., 0]) + tf.square(gxy[..., 1])
        )
        gxy = tf.expand_dims(gxy, axis=-1)
        theta = tf.expand_dims(theta, axis=-1)
        edge_before_thresh = None

        for angle in self.angles_names:
            call_, range_ = self.__getattribute__(f'call_range_{angle}')
            bool_ang = call_(
                tf.math.greater_equal(
                    theta, range_[0]
                ),
                tf.math.less(
                    theta, range_[1]
                )
            )
            ang_image = tf.where(
                bool_ang, gxy, 0.0
            )
            ang_image_pad = self.zero_pad(ang_image)

            kernel = self.__getattribute__(f'local_max_kernel_{angle}')
            kernel = tf.reshape(
                tf.constant(
                    kernel, tf.float32
                ), [3, 3, 1]
            )
            max_pool_ang = tf.nn.dilation2d(
                ang_image_pad, kernel, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1)
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
                edge_before_thresh, self.threshold_setup['max_val']
            ), 1.0, 0.0
        )
        edge_week = tf.where(
            tf.math.logical_and(
                tf.math.greater_equal(
                    edge_before_thresh, self.threshold_setup['min_val']
                ),
                tf.math.less(
                    edge_before_thresh, self.threshold_setup['max_val']
                )
            ), 1.0, 0.0
        )

        connected = None

        for i in range(self.hysteresis_setup['connection_iterations']):
            if connected is None:
                connected_sure = tf.nn.dilation2d(
                    edge_sure, self.hysteresis_kernel, (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1)
                ) - 1

                connected_week = tf.nn.dilation2d(
                    edge_week, self.hysteresis_kernel, (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1)
                ) - 1
                connected = tf.where(
                    tf.math.greater_equal(
                        (connected_sure * edge_week) + (connected_week * edge_sure), 1.0
                    ), 1.0, 0.0
                )
                continue

            prev_connected = connected
            connected = tf.nn.dilation2d(
                connected, self.hysteresis_kernel, (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1)
            ) - 1

            connected = tf.where(
                tf.math.greater_equal(
                    (connected * edge_week) + (connected * edge_sure), 1.0
                ), 1.0, 0.0
            )
            if tf.math.reduce_max(connected - prev_connected) == 0:
                break

        edge = tf.where(
            tf.math.greater_equal(
                connected + edge_sure, 1.0
            ), 255.0, 0.0
        )
        return edge
