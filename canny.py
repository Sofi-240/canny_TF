import tensorflow as tf

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


def gaussian_setup(kernel_size, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    normal = 1 / (2.0 * PI * (sigma ** 2))
    kernel = tf.exp(
        -((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2))
    ) * normal

    kernel = kernel / tf.reduce_sum(kernel)

    out = tf.reshape(
        kernel, shape=(kernel_size, kernel_size, 1, 1), name='gaussian_kernel'
    )
    return out


def sobel_setup():
    h = tf.reshape(
        tf.constant(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], tf.float32
        ), shape=(3, 3, 1, 1), name='H'
    )
    v = tf.reshape(
        tf.constant(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], tf.float32
        ), shape=(3, 3, 1, 1), name='V'
    )
    out = tf.concat(
        (h, v), axis=-1, name='sobel_kernel'
    )
    return out


def localmax_setup():
    ang_0_kernel = tf.reshape(
        tf.constant(
            [
                [-float('inf'), -float('inf'), -float('inf')],
                [0.0, 0.0, 0.0],
                [-float('inf'), -float('inf'), -float('inf')]
            ], dtype=tf.float32
        ), shape=(3, 3, 1), name='ang_0_kernel'
    )
    ang_45_kernel = tf.reshape(
        tf.constant(
            [
                [-float('inf'), -float('inf'), 0.0],
                [-float('inf'), 0.0, -float('inf')],
                [0.0, -float('inf'), -float('inf')]
            ], dtype=tf.float32
        ), shape=(3, 3, 1), name='ang_45_kernel'
    )
    ang_90_kernel = tf.reshape(
        tf.constant(
            [
                [-float('inf'), 0.0, -float('inf')],
                [-float('inf'), 0.0, -float('inf')],
                [-float('inf'), 0.0, -float('inf')]
            ], dtype=tf.float32
        ), shape=(3, 3, 1), name='ang_90_kernel'
    )
    ang_135_kernel = tf.reshape(
        tf.constant(
            [
                [0.0, -float('inf'), -float('inf')],
                [-float('inf'), 0.0, -float('inf')],
                [-float('inf'), -float('inf'), 0.0]
            ], dtype=tf.float32
        ), shape=(3, 3, 1), name='ang_90_kernel'
    )
    angles_setup = [
        (tf.math.logical_or, (157.5, 22.5), ang_0_kernel),
        (tf.math.logical_and, (22.5, 67.5), ang_45_kernel),
        (tf.math.logical_and, (67.5, 112.5), ang_90_kernel),
        (tf.math.logical_and, (112.5, 157.5), ang_135_kernel)
    ]
    return angles_setup


def pad(images, b=None, h=None, w=None, d=None, **kwargs):
    paddings = []
    for arg in [b, h, w, d]:
        arg = arg if arg is not None else [0, 0]
        arg = [arg, arg] if issubclass(type(arg), int) else list(arg)
        paddings.append(arg)

    paddings = tf.constant(paddings, dtype=tf.int32)

    padded = tf.pad(
        images, paddings, **kwargs
    )
    return padded


@tf.function
def canny_edge(images, sigma=None, kernel_size=5, min_val=50, max_val=100, connection_iterations=20):
    X = tf.identity(images, name='X') if tf.is_tensor(images) else tf.convert_to_tensor(images, name='X')
    n_dim = len(X.shape)
    if n_dim < 2:
        raise ValueError(
            f'expected for 2/3/4D tensor but got {n_dim} tensor'
        )
    X = tf.expand_dims(X, axis=-1, name='X') if n_dim == 2 else X
    X = tf.expand_dims(X, axis=0, name='X') if n_dim <= 3 else X

    X = tf.cast(X, dtype=tf.float32, name='X')

    gaussian_kernel = gaussian_setup(kernel_size, sigma)
    sobel_kernel = sobel_setup()
    angle_cond_rng_kernel = localmax_setup()
    hysteresis_kernel = tf.ones(shape=(5, 5, 1), dtype=tf.float32)

    with tf.name_scope('canny_edge'):
        Xg = tf.nn.convolution(
            pad(X, h=kernel_size // 2, w=kernel_size // 2, mode='SYMMETRIC', name='X_pad'),
            gaussian_kernel, padding='VALID', name='Xg'
        )

        Gxy = tf.nn.convolution(
            pad(Xg, h=1, w=1, constant_values=0.0, name='Xg_pad'),
            sobel_kernel, padding='VALID', name='Gxy'
        )

        gx, gy = tf.split(Gxy, [1, 1], axis=-1)

        theta = ((tf.atan2(gx, gy, name='theta') * 180 / PI) + 90) % 180
        Gxy = tf.sqrt((gx ** 2) + (gy ** 2), name='Gxy')
        edge_before_thresh = None

        angles_setup = [
            (tf.math.logical_or, (157.5, 22.5), ang_0_kernel),
            (tf.math.logical_and, (22.5, 67.5), ang_45_kernel),
            (tf.math.logical_and, (67.5, 112.5), ang_90_kernel),
            (tf.math.logical_and, (112.5, 157.5), ang_135_kernel)
        ]

        for call_, rng, kernel in angle_cond_rng_kernel:
            ang_image = tf.where(
                call_(
                    tf.math.greater_equal(theta, rng[0]),
                    tf.math.less(theta, rng[1])
                ), Gxy, 0.0
            )
            max_pool_ang = tf.nn.dilation2d(
                pad(ang_image, h=1, w=1, constant_values=0.0),
                kernel, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1)
            )

            is_max = tf.where(
                tf.math.equal(max_pool_ang, ang_image), ang_image, 0.0
            )
            if edge_before_thresh is None:
                edge_before_thresh = is_max
            else:
                edge_before_thresh = edge_before_thresh + is_max

        edge_sure = tf.where(
            tf.math.greater_equal(edge_before_thresh, max_val), 1.0, 0.0, name='edge_sure'
        )
        edge_week = tf.where(
            tf.math.logical_and(
                tf.math.greater_equal(edge_before_thresh, min_val),
                tf.math.less(edge_before_thresh, max_val)
            ), 1.0, 0.0, name='edge_week'
        )

        connected = tf.add(edge_sure, edge_week, name='connected')

        for _ in tf.range(connection_iterations):

            prev_connected = tf.identity(connected, name='prev_connected')
            connected = tf.nn.dilation2d(
                prev_connected, hysteresis_kernel,
                (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1), name='connected'
            ) - 1

            connected = tf.where(
                tf.math.greater_equal(
                    (connected * edge_week) + (connected * edge_sure), 1.0
                ), 1.0, 0.0, name='connected'
            )
            if tf.math.reduce_max(connected - prev_connected) == 0:
                break

        edge = tf.where(
            tf.math.greater_equal(
                connected + edge_sure, 1.0
            ), 255.0, 0.0, name='edge'
        )
        return edge
