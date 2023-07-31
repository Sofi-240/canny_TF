import tensorflow as tf

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


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


def noise_reduction(X, kernel_size, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    normal = 1 / (2.0 * PI * (sigma ** 2))
    kernel = tf.exp(
        -((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2))
    ) * normal

    kernel = kernel / tf.reduce_sum(kernel)

    kernel = tf.reshape(
        kernel, shape=(kernel_size, kernel_size, 1, 1), name='gaussian_kernel'
    )
    with tf.name_scope('noise_reduction'):
        X_pad = pad(X, h=kernel_size // 2, w=kernel_size // 2, mode='SYMMETRIC', name='X_pad')
        Xg = tf.nn.convolution(X_pad, kernel, padding='VALID', name='Xg')
        return Xg


def gradient_calculation(X):
    sobel_kernel = tf.constant(
        [
            [[[-1, -1]], [[0, -2]], [[1, -1]]],
            [[[-2, 0]], [[0, 0]], [[2, 0]]],
            [[[-1, 1]], [[0, 2]], [[1, 1]]]
        ], dtype=tf.float32, name='sobel_kernel'
    )
    with tf.name_scope('gradient_calculation'):
        X_pad = pad(X, h=1, w=1, constant_values=0.0, name='X_pad')
        Gxy = tf.nn.convolution(X_pad, sobel_kernel, padding='VALID', name='Gxy')
        gx, gy = tf.split(Gxy, [1, 1], axis=-1)

        theta = ((tf.atan2(gx, gy, name='theta') * 180 / PI) + 90) % 180
        Gxy = tf.sqrt((gx ** 2) + (gy ** 2), name='Gxy')
        return Gxy, theta


def non_maximum_suppression(Gxy, theta):
    ang_0_kernel = tf.constant(
        [
            [[-float('inf')], [-float('inf')], [-float('inf')]],
            [[0.0], [0.0], [0.0]],
            [[-float('inf')], [-float('inf')], [-float('inf')]]
        ], dtype=tf.float32, name='ang_0_kernel'
    )
    ang_45_kernel = tf.constant(
        [
            [[-float('inf')], [-float('inf')], [0.0]],
            [[-float('inf')], [0.0], [-float('inf')]],
            [[0.0], [-float('inf')], [-float('inf')]]
        ], dtype=tf.float32, name='ang_45_kernel'
    )
    ang_90_kernel = tf.constant(
        [
            [[-float('inf')], [0.0], [-float('inf')]],
            [[-float('inf')], [0.0], [-float('inf')]],
            [[-float('inf')], [0.0], [-float('inf')]]
        ], dtype=tf.float32, name='ang_90_kernel'
    )
    ang_135_kernel = tf.constant(
        [
            [[0.0], [-float('inf')], [-float('inf')]],
            [[-float('inf')], [0.0], [-float('inf')]],
            [[-float('inf')], [-float('inf')], [0.0]]
        ], dtype=tf.float32, name='ang_90_kernel'
    )
    angle_kernel = tf.concat(
        [tf.expand_dims(kernel, axis=0) for kernel in \
         [ang_0_kernel, ang_45_kernel, ang_90_kernel, ang_135_kernel]],
        axis=0, name='angle_kernel'
    )
    angle_rng = tf.constant(
        [[157.5, 22.5],
         [22.5, 67.5],
         [67.5, 112.5],
         [112.5, 157.5]], dtype=tf.float32, name='angle_rng'
    )

    edge_before_thresh = tf.zeros_like(Gxy, name='edge_before_thresh')

    with tf.name_scope('non_maximum_suppression'):
        for i in tf.range(4):
            kernel = angle_kernel[i, ...]
            rng = angle_rng[i]
            if i == 0:
                ang_image = tf.where(
                    tf.math.logical_or(
                        tf.math.greater_equal(theta, rng[0]),
                        tf.math.less_equal(theta, rng[1])
                    ), Gxy, 0.0
                )
            else:
                ang_image = tf.where(
                    tf.math.logical_and(
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
            edge_before_thresh = edge_before_thresh + is_max
        return edge_before_thresh


def double_thresholding(X, min_val=50, max_val=100):
    with tf.name_scope('double_thresholding'):
        edge_sure = tf.where(
            tf.math.greater_equal(X, max_val), 1.0, 0.0, name='edge_sure'
        )
        edge_week = tf.where(
            tf.math.logical_and(
                tf.math.greater_equal(X, min_val),
                tf.math.less(X, max_val)
            ), 1.0, 0.0, name='edge_week'
        )
        return edge_sure, edge_week


def hysteresis_tracking(edge_sure, edge_week, iterations=20, alg='connected'):
    with tf.name_scope('hysteresis_tracking'):
        if alg == 'connected':
            return connected_alg(edge_sure, edge_week, iterations)
        else:
            return None


def connected_alg(edge_sure, edge_week, iterations=20):
    hysteresis_kernel = tf.ones(shape=(5, 5, 1), dtype=tf.float32)

    connected = tf.add(edge_sure, edge_week, name='connected')
    # TODO: bw open?  ---> dilation2d(erosion2d + edge) - edge ---> where >= 1 else 0
    with tf.name_scope('connected_alg'):
        for _ in tf.range(iterations):

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


@tf.function
def canny_edge(images, sigma=None, kernel_size=5,
               min_val=50, max_val=100, hysteresis_alg='connected', connection_iterations=20):
    X = tf.identity(images, name='X') if tf.is_tensor(images) else tf.convert_to_tensor(images, name='X')
    n_dim = len(X.shape)
    if n_dim < 2:
        raise ValueError(
            f'expected for 2/3/4D tensor but got {n_dim} tensor'
        )
    X = tf.expand_dims(X, axis=-1, name='X') if n_dim == 2 else X
    X = tf.expand_dims(X, axis=0, name='X') if n_dim <= 3 else X

    X = tf.cast(X, dtype=tf.float32, name='X')
    with tf.name_scope('canny_edge'):
        Xg = noise_reduction(
            X, kernel_size=kernel_size, sigma=sigma
        )
        Gxy, theta = gradient_calculation(Xg)
        edge_before_thresh = non_maximum_suppression(
            Gxy, theta
        )
        edge_sure, edge_week = double_thresholding(
            edge_before_thresh, min_val=min_val, max_val=max_val
        )
        edge = hysteresis_tracking(
            edge_sure, edge_week, iterations=connection_iterations, alg=hysteresis_alg
        )
        return edge

