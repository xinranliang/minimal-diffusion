import numpy as np
import cv2

def rgb_to_grayscale(image):
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


def test():
    color_image = cv2.imread("/home/xinranliang/projects/minimal-diffusion/logs/cifar10_color/2022-12-01/samples/UNet_cifar10-1000_steps-250-sampling_steps-class_condn_False.png")
    gray_image = rgb_to_grayscale(color_image)
    cv2.imwrite("color.png", color_image)
    cv2.imwrite("grayscale.png", gray_image)
    return 

if __name__ == "__main__":
    test()