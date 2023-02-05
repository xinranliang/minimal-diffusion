import numpy as np


def rgb_to_gray(image):
    """
    Function to convert RGB channel images to grayscale images, still preserve shape/dimension

    Input: height x width x 3
    Output: height x width x 3

    """

    gray_image = np.zeros(image.shape)
    R = np.array(image[:, :, 0])
    G = np.array(image[:, :, 1])
    B = np.array(image[:, :, 2])

    avg_channel = R * 0.299 + G * 0.587 + B * 0.114

    for i in range(3):
        gray_image[:,:,i] = avg_channel
           
    return gray_image