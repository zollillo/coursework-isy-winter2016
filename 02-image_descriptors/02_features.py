import cv2

cap = cv2.VideoCapture(0)
# Resize frame
cap.set(3, 640)  # set frame width
cap.set(4, 480)  # set frame height
cv2.namedWindow('Interactive Systems: Towards AR Tracking')
while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # YOUR CODE HERE
    # ==============

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert from BGR->Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Find key points in frame
    kp = sift.detect(gray, None)

    # Visualize the key points by drawing small circles at key points' locations.
    # Use flags to indicate the size and orientation of each key point.
    kp_viz = cv2.drawKeypoints(gray, kp, None, color=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the resulting video stream key point visualization
    cv2.imshow('Interactive Systems: Towards AR Tracking', kp_viz)
    cv2.moveWindow('Interactive Systems: Towards AR Tracking', 0, 0)

    # Wait for keyboard event
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print 'You pressed %s. Quitting demo...' % chr(k)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
