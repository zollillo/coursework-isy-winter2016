import numpy as np
import cv2
import glob

# code taken from: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html


# Modifications made are based on following resources:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# http://www.janeriksolem.net/2014/05/how-to-calibrate-camera-with-opencv-and.html
# https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
pattern_size = (7, 7)
pattern_points = np.zeros((np.product(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.


cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set frame width
cap.set(4, 480)  # set frame height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    # Wait for keyboard event
    k = cv2.waitKey(1) & 0xFF

    # Wait for 'c' to use current frame as a test pattern for camera calibration.
    if k == ord('c'):
        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # Show info if no chessboard corners were found.
        if not found:
            print('Chessboard not found!')
            continue

        # If found, add object points, image points (after refining them).
        if found:
            print 'Chessboard found!'
            obj_points.append(pattern_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(frame, pattern_size, corners2, found)
            cv2.imshow('img', img)

            # Calculate camera distortion
            rms, mtx, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None,
                                                                     None)

            print '\nroot mean square (RMS) projection error:', rms
            print 'camera matrix:\n', mtx
            print 'distortion coefficients:', dist_coefs.ravel()

    # Wait for 'q' to exit
    elif k == ord('q'):
        break
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
#
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
#
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
#         img = cv2.imread('images/left12.jpg')
#         h, w = img.shape[:2]
#         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#
#         # undistort
#         dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
#         # crop the image
#         x, y, w, h = roi
#         dst = dst[y:y + h, x:x + w]
#         cv2.imwrite('calibresult.png', dst)
#
# tot_error = 0
# mean_error = 0
# for i in xrange(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     print error
#     tot_error += error
#     mean_error += error
#
# print "mean error: ", mean_error / len(objpoints), " total error: ", tot_error

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
