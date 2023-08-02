import tensorflow as tf
import collections

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


def pad(X, b=None, h=None, w=None, d=None, **kwargs):
    n_dim = len(X.get_shape())
    assert n_dim == 4
    if not b and not h and not w and not d: return X
    paddings = []
    for arg in [b, h, w, d]:
        arg = arg if arg is not None else [0, 0]
        arg = [arg, arg] if issubclass(type(arg), int) else list(arg)
        paddings.append(arg)

    paddings = tf.constant(paddings, dtype=tf.int32)

    padded = tf.pad(
        X, paddings, **kwargs
    )
    return padded


def noise_reduction(X, kernel_size, sigma=None):
    n_dim = len(X.get_shape())
    assert n_dim == 4
    assert kernel_size % 2 != 0 and kernel_size > 2

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
        X_pad = pad(X, h=kernel_size // 2, w=kernel_size // 2, constant_values=0.0, name='X_pad')
        Xg = tf.nn.convolution(X_pad, kernel, padding='VALID', name='Xg')
        return Xg


def gradient_calculation(X):
    n_dim = len(X.get_shape())
    assert n_dim == 4
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
    gxy_shape = Gxy.get_shape()
    theta_shape = theta.get_shape()
    assert len(gxy_shape) == 4 and len(theta_shape) == 4

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
        for i in tf.range(angle_rng.shape[0]):
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


def hysteresis_tracking(edge_sure, edge_week, alg='dilation', con=None, iterations=None):
    assert alg == 'connection' or alg == 'dilation'
    with tf.name_scope('hysteresis_tracking'):
        if alg == 'dilation':
            return dilation_tracking(edge_sure, edge_week, con=con, iterations=iterations)
        else:
            return connection_tracking(edge_sure, edge_week, con=con, iterations=iterations)


def dilation_tracking(edge_sure, edge_week, iterations=20, con=5):
    hysteresis_kernel = tf.ones(shape=(con, con, 1), dtype=tf.float32)

    with tf.name_scope('dilation_alg'):
        dil = tf.nn.dilation2d(
            edge_sure, hysteresis_kernel, (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1)
        )
        connected = (dil * edge_week) + edge_sure
        for _ in tf.range(iterations):
            prev_connected = tf.identity(connected, name='prev_connected')
            dil = tf.nn.dilation2d(
                connected, hysteresis_kernel, (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1)
            )
            connected = (dil * edge_week) + edge_sure

            if tf.math.reduce_max(connected - prev_connected) == 0:
                break

        return tf.identity(connected, name='edge')


def connection_tracking(edge_sure, edge_week, con=7, iterations=None):
    B, H, W, _ = edge_sure.shape
    ax = tf.range(
        -con // 2 + 1, (con // 2) + 1, dtype=tf.int64
    )
    con_kernel = tf.stack(
        tf.meshgrid(ax, ax), axis=-1
    )
    con_kernel = tf.reshape(
        con_kernel, shape=(1, con ** 2, 2)
    )
    con_up, _, con_down = tf.split(con_kernel, [(con ** 2) // 2, 1, (con ** 2) // 2], axis=1)
    con_kernel = tf.concat((con_up, con_down), axis=1, name='con_kernel')
    repeats = con_kernel.shape[1]

    def make_neighbor(cords, name=None):
        name = name if name is not None else 'neighbor'
        b, yx, _ = tf.split(
            cords, [1, 2, 1], axis=1
        )
        yx = yx[:, tf.newaxis, ...]

        yx = yx + con_kernel

        b = tf.repeat(
            b[:, tf.newaxis, ...], repeats=repeats, axis=1
        )

        y, x = tf.split(yx, [1, 1], axis=-1)
        y = tf.reshape(y, shape=(-1,))
        y = tf.where(tf.logical_or(tf.math.greater(y, H - 1), tf.math.less(y, 0)), 0, y)
        y = tf.reshape(y, shape=(-1, repeats, 1))

        x = tf.reshape(x, shape=(-1,))
        x = tf.where(tf.logical_or(tf.math.greater(x, W - 1), tf.math.less(x, 0)), 0, x)
        x = tf.reshape(x, shape=(-1, repeats, 1))

        neighbor = tf.concat(
            (b, y, x), axis=-1, name=name
        )
        neighbor = tf.reshape(neighbor, shape=(-1, 3))
        return tf.pad(
            neighbor, paddings=tf.constant([[0, 0], [0, 1]]), constant_values=0, name=name
        )

    output = tf.zeros_like(edge_sure)
    connected_index = tf.where(tf.math.equal(edge_sure, 1.0))

    with tf.name_scope('connection_tracking'):
        def one_iter(out, con_index, week):
            n = con_index.get_shape()[0]
            out = tf.tensor_scatter_nd_update(
                out, con_index, tf.ones((n,), dtype=tf.float32)
            )
            week = week * (1.0 - out)
            index_lookup = make_neighbor(con_index)
            n = index_lookup.get_shape()[0]
            unique = tf.scatter_nd(
                index_lookup, tf.ones((n,), dtype=tf.float32), shape=out.shape
            )

            con_index = tf.where(
                tf.math.equal(unique * week, 1.0)
            )
            return out, con_index, week

        def check(out, con_index, week):
            n = con_index.get_shape()[0]
            if not n or n == 0:
                return False
            return True

        loop_params = collections.namedtuple('loop_params', 'out, con_index, week')
        loop_vars = loop_params(output, connected_index, edge_week)

        output, connected_index, edge_week = tf.while_loop(check, one_iter, loop_vars=loop_vars,
                                                           maximum_iterations=iterations)
        return tf.identity(output, name='edge')


def canny_edge(images,
               sigma=None,
               kernel_size=5,
               min_val=50,
               max_val=100,
               hysteresis_tracking_alg='dilation',
               tracking_con=5,
               tracking_iterations=20):
    X = tf.identity(images, name='X') if tf.is_tensor(images) else tf.convert_to_tensor(images, name='X')

    shape_ = X.get_shape()
    n_dim_ = len(shape_)
    d_type = X.dtype

    if n_dim_ < 2:
        raise ValueError(
            f'expected for 2/3/4D tensor but got {n_dim_}D'
        )

    X = tf.reshape(X, shape=(1, shape_[0], shape_[1], 1), name='X') if n_dim_ == 2 else X
    if n_dim_ == 3:
        X = tf.expand_dims(X, axis=-1, name='X') if shape_[-1] > 3 else tf.expand_dims(X, axis=0, name='X')

    if X.shape[-1] != 3 and X.shape[-1] != 1:
        raise ValueError(
            f'expected feature dim with size of 3 for RGB image and size of 1 for gray scale image'
        )

    if shape_[-1] == 3:
        X = tf.image.rgb_to_grayscale(X, name='X')

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
            edge_sure, edge_week, alg=hysteresis_tracking_alg, con=tracking_con, iterations=tracking_iterations
        )

    if d_type == 'uint8':
        edge = edge * 255.0

    edge = tf.cast(edge, dtype=d_type, name='edge')
    return edge
