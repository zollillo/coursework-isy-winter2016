import numpy as np
import cv2
import math
import sys
from glob import glob
from ImageStitcher import *


############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
img_names = glob('images/pano*.jpg')
for fname in img_names:
    print 'processing %s... ' % fname

img1 = cv2.imread('images/pano6.jpg')
img2 = cv2.imread('images/pano5.jpg')
img3 = cv2.imread('images/pano4.jpg')
# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher([img1, img2, img3]) # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

# if not matchlist:
if not imageStitcher:
    print "We have not enough matching keypoints to create a panorama"
else:
    # YOUR CODE HERE
    scaling_factor = 0.5
    matchlist = cv2.resize(matchlist, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    result = cv2.resize(result, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # output all matching images
    cv2.imshow('Matching Images', matchlist)
    # output result
    cv2.imshow('Stitched Panorama', result)
    # Note: if necessary resize the image
    cv2.waitKey(0)

