# Working with OpenCV and Python

The examples in this repository are some of the results of exercises from the university course _Interactive Systems_ taught by [Prof. Dr.-Ing. Kristian Hildebrand](http://hildebrand.beuth-hochschule.de/) at Beuth University of Applied Sciences Berlin in the winter term 2016.

To learn about some common problems in _Computer Vision_ we used [OpenCV (version 3.1.0)](http://docs.opencv.org/3.1.0/d1/dfb/intro.html) and [Python 2.7 in the form of an Anaconda Python distribution](https://docs.continuum.io/).

* `/camera-calibration/camera-calibration.py` is based on the [Camera Calibration](http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html) tutorial provided as part of the [OpenCV Python Tutorials](http://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html). However, instead of using multiple test pattern images, we were asked to perform the calibration of our camera by capturing a live video stream and upon keystroke using the current frame as test pattern for calibrating the camera.  

* `/feature-detection/features.py` is an example to use the SIFT (Scale-Invariant Feature Transform) algorithm to find and visualize the keypoints in a live video stream. Based on the tutorial [Introduction to SIFT](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html).

* `/image-retrieval/image-retrieval.py` is an example of a simple content-based image search using [SIFT](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html) keypoints and descriptors. _(Note: images in `/image-retrieval/db` are copyrighted by their respective copyright owners.)_
