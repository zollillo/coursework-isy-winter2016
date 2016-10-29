import numpy as np
import cv2
import time
from gaussian_kernel_calculator import *

# (3) Please implement a convolution on grayscale image.
# use g = GaussianKernelCalculator(1.0, 5) as kernel, try changing sizes of the kernel
# and measure time of execution using

# start_time = time.time()
# imgConv2d = convolution2d(g.kernel2d, imgGray)
# print "convolution2d time:", time.time() - start_time

# start_time = time.time()
# imgConv1d = convolution1d(g.kernel1d, imgGray)
# print "convolution1d time:", time.time() - start_time

# you can save the image like this:
# cv2.imwrite(filename, image)


# Implementation is based on the the following resource:
# http://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# Further references:
# http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering


def convolution2d(kernel2d, source):
    source_rows, source_columns = source.shape[:2]
    kernel_rows, kernel_columns = kernel2d.shape[:2]

    # Replicate pixels at border of source image by applying padding according to kernel size.
    # This operation makes sure that the convolved output will match the original size of the source image.
    padding = (kernel_columns - 1) / 2
    source = cv2.copyMakeBorder(source, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    # Convert source image to floating point values in range [0, 1]
    source = np.array(source, dtype='f') / 255

    # Create a new image for the output with dimensions of original source
    output = np.zeros((source_rows, source_columns), dtype='f')

    # Traverse the source image pixel by pixel
    for y in np.arange(padding, source_rows + padding):
        for x in np.arange(padding, source_columns + padding):
            # Use numpy indexing to find the region of interest (ROI) of the current (x,y) position in the source image
            # in order to place the center element of the kernel over the source pixel.
            roi = source[y - padding:y + padding + 1, x - padding: x + padding + 1]

            # Compute new value (i.e. perform convolution) by calculating the sum of the element-wise product
            # of kernel2d matrix and ROI matrix.
            new_pixel_value = (kernel2d * roi).sum()

            # Assign new value at indices of output array that correspond to (x,y) coordinate of source image.
            output[y - padding, x - padding] = new_pixel_value

    # Convert output image to array of unsigned int values in range [0, 255]
    output = (output * 255).astype('uint8')
    # print 'Output matrix =\n', output
    return output


def convolution1d(kernel1d, source):
    rows, columns = source.shape[:2]
    pass


g = GaussianKernelCalculator(1.0, 5)
# print 'kernel1d =', g.kernel1d
# print 'kernel2d =\n', g.kernel2d

# Load image from file as a gray scale image
# Question: What is the difference between loading in gray scale mode and performing color conversion?
# - gray scale mode
# imgGray = cv2.imread('Cute-Red-Squirrel.jpeg', 0)
# - convert from BGR -> Gray
imgBGR = cv2.imread('parrot.jpeg')
imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)


start_time = time.time()
imgConv2d = convolution2d(g.kernel2d, imgGray)
print "convolution2d time:", time.time() - start_time
# cv2.imshow('Original', imgGray)
# cv2.imshow('Convolved 2D', imgConv2d)

# while True:
#     k = cv2.waitKey(0)
#     if k == ord('q'):
#         break
#
# cv2.destroyAllWindows()
