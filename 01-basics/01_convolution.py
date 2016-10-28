import numpy as np
import cv2
import time
from gaussian_kernel_calculator import *

# (3) Please implement a convolution on grayscale image.
# use g = GaussianKernelCalculator(1.0, 5) as kernel, try changing sizes of the kernel
# and measure time of execution using

g = GaussianKernelCalculator(1.0, 5)

imgGray = cv2.imgread('Cute-Red-Squirrel.jpeg', 0)
# Convert image to floating point values in range [0, 1]
imgGray = np.array(imgGray, dtype='f') / 255
cv2.imshow('Convolution', imgGray)



# start_time = time.time()
# imgConv2d = convolution2d(g.kernel2d, imgGray)
# print "convolution2d time:", time.time() - start_time

# start_time = time.time()
# imgConv1d = convolution1d(g.kernel1d, imgGray)
# print "convolution1d time:", time.time() - start_time


# you can save the image like this:
# cv2.imwrite(filename, image)

def convolution2d(kernel2d, img):
    pass

def convolution1d(kernel1d, img):
    pass