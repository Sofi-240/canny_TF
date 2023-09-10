# Canny Edge Detection with TensorFlow

Canny Edge Detection is a widely used image processing technique for identifying edges within images. 
This project provides an implementation of the Canny Edge Detection algorithm using TensorFlow, a popular deep learning framework. 
With this implementation, you can quickly and easily apply edge detection to your images.

![Canny Edge Detection Example](demo_gif_full.gif)

## Features

- Fast and efficient Canny edge detection using TensorFlow.
- Adjustable parameters for sigma, thresholding, and tracking.
- Supports both grayscale and RGB images.

## Prerequisites

Before you get started, make sure you have the following dependencies installed:

- Python 3.7 or higher
- TensorFlow
- Dataclasses (for Python 3.6 users)

## Usage

1. Import the Canny class and other necessary modules.
`import tensorflow as tf`
`from canny import Canny`

2. Create an instance of the Canny class with your desired parameters. 
You can customize the following parameters:

- sigma: Controls the smoothing strength of the Gaussian filter (default: 0.8).
- threshold_min: Minimum threshold for edge detection (default: 50).
- threshold_max: Maximum threshold for edge detection (default: 80).
- tracking_con: Size of the dilation kernel for edge tracking (default: 5).
- tracking_iterations: Number of iterations for edge tracking (default: 5).

`canny = Canny(sigma=0.8, threshold_min=50, threshold_max=80, tracking_con=5, tracking_iterations=5)`

3. Load and preprocess your image using TensorFlow. Ensure that your image is either grayscale or RGB.

4. Apply Canny edge detection to the image:
`edges = canny(image)`

## Customization
Feel free to experiment with different parameter values when creating the Canny object to fine-tune edge detection for your specific use case.

## Acknowledgments
This project is based on the Canny Edge Detection algorithm originally developed by John Canny. The TensorFlow implementation allows for seamless integration into deep learning workflows.

