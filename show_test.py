import cv2
from matplotlib import pyplot as plt
from canny import canny_edge
import matplotlib

matplotlib.use("Qt5Agg")

image = cv2.imread('test_img.jpg')
image = cv2.cvtColor(
    image, cv2.COLOR_BGR2GRAY
)
edges_opencv = cv2.Canny(
    image, 50, 100
)

edge_tf = canny_edge(image, max_val=100, min_val=50, connection_iterations=5, sigma=0.5)


fig, ax = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(wspace=0)
ax[0].imshow(image, cmap='gray')
ax[1].imshow(edges_opencv, cmap='gray')
ax[2].imshow(edge_tf.numpy()[0].astype('uint8'), cmap='gray')