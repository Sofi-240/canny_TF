if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from canny import canny_edge
    import cv2
    import matplotlib

    matplotlib.use("Qt5Agg")

    image = cv2.imread('V2_2.jpg')[..., ::-1]

    edges_connection = canny_edge(image,
                                  sigma=0.8,
                                  kernel_size=5,
                                  min_val=50,
                                  max_val=100,
                                  hysteresis_tracking_alg='connection',
                                  tracking_con=11,
                                  tracking_iterations=None)

    edges_dilation = canny_edge(image,
                                sigma=0.8,
                                kernel_size=5,
                                min_val=50,
                                max_val=100,
                                hysteresis_tracking_alg='dilation',
                                tracking_con=5,
                                tracking_iterations=20)

    fig, _ = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.axes[0].imshow(image)
    fig.axes[0].set_title('Original image')

    fig.axes[1].imshow(edges_connection[0, ...].numpy(), cmap='gray')
    fig.axes[1].set_title('Edge image with connection Alg.')

    fig.axes[2].imshow(edges_dilation[0, ...].numpy(), cmap='gray')
    fig.axes[2].set_title('Edge image with dilation Alg.')
