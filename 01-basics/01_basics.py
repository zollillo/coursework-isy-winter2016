import numpy as np
import cv2
import math
import time
from gaussian_kernel_calculator import *


######################################################################
# IMPORTANT: Please make yourself comfortable with numpy and python:
# e.g. https://www.stavros.io/tutorials/python/
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Note: data types are important for numpy and opencv
# most of the time we'll use np.float32 as arrays
# e.g. np.float32([0.1,0.1]) equal np.array([1, 2, 3], dtype='f')

# A1. Numpy and Linear Algebra

# (1) Define Matrix M and column vectors vec0 ... vec3 in numpy
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [0, 2, 2]], dtype='f')
vec0 = np.array([1, 1, 0], dtype='f').reshape(3, 1)
vec1 = np.array([-1, 2, 5], dtype='f').reshape(3, 1)
vec2 = np.array([1, 2, 3, 4, 5], dtype='f').reshape(5, 1)
vec3 = np.array([-1, 9, 5, 3, 1], dtype='f').reshape(5, 1)

# Print to console to check the results
print 'M ='
print M
print 'Shape of M is', M.shape, '\n'

print 'vec0 ='
print vec0
print 'Shape of vec0 is', vec0.shape, '\n'

print 'vec1 ='
print vec1
print 'Shape of vec1 is', vec1.shape, '\n'

print 'vec2='
print vec2
print 'Shape of vec2 is', vec2.shape, '\n'

print 'vec3='
print vec3
print 'Shape of vec3 is', vec3.shape, '\n'


# (2) Please implement a dot product and magnitude function for arbitrary vectors
# yourself using numpy float vectors and use them given the vectors
# do not use the dot product given by numpy
#
# usage: dot(vec0,vec1) and dot(vec2, vec3), norm(v0-v1)
def dot(v0, v1):
    pass

def norm(v):
    pass

# (3) compute vec0^T vec1 M vec0 using numpy operations
# be aware of what is a column and a row vector

######################################################################
# A2. OpenCV and Transformation and Computer Vision Basic

# (1) read in the image Lenna.png using opencv in gray scale and in color
# and display it NEXT to each other (see result image)
# Note here: the final image displayed must have 3 color channels
#            So you need to copy the gray image values in the color channels
#            of a new image. You can get the size (shape) of an image with rows, cols = img.shape[:2]

# why Lenna? https://de.wikipedia.org/wiki/Lena_(Testbild)

# (2) Now shift both images by half (translation in x) it rotate the colored image by 30 degrees using OpenCV transformation functions
# + do one of the operations on keypress (t - translate, r - rotate, 'q' - quit using cv::warpAffine
# http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
# Tip: you need to define a transformation Matrix M
# see result image


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

def convolution2d(kernel2d, img):
    pass

def convolution1d(kernel1d, img):
    pass


