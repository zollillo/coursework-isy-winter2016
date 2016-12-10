import numpy as np
import cv2
import glob
from Queue import PriorityQueue
import math

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
# Resources:
# http://stackoverflow.com/a/32142625
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
def distance(a, b):
    return np.linalg.norm(a - b)


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    # YOUR CODE HERE
    for r in range(0, h, keypointSize):
        for c in range(0, w, keypointSize):
            keypoints.append(cv2.KeyPoint(c, r, keypointSize))
    return keypoints


# 1. preprocessing and load
images = glob.glob('./db/*/*.jpg')


# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)


# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.

# YOUR CODE HERE
# Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

for img_file in images:
    # print 'processing %s... ' % img_file
    # Read image
    img = cv2.imread(img_file)

    # Convert from BGR->Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute SIFT descriptors
    kp, des = sift.compute(gray, keypoints)

    # Store a tuple containing the image and its descriptor
    descriptors.append((img_file, des))


# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE
# Create priority queue
q = PriorityQueue()

# Load query image
query_img = cv2.imread('./db/query_face.jpg')
# query_img = cv2.imread('./images/db/query_car.jpg')
# query_img = cv2.imread('./images/db/query_flower.jpg')

# Convert query image from BGR->Gray
query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

# Extract descriptor of query image
query_kp, query_des = sift.compute(query_gray, keypoints)

# Compare features of query image to all features of the images from our db
# using the implemented distance function.
# Store the output of the distance function and the associated image in
# a priority queue so as to later pull out the images based on priority order,
# i.e. in the order of smallest distance.
# In other words: the most similar images are of higher priority.
for p, d in descriptors:
    q.put((distance(query_des, d), p))


# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE
# HACK: A kind of cumbersome approach to display all the images
# that are stored in our priority queue in one image.

# Original images in db are 256x256, we want to display them half their size
# arranged in a 2xn grid where n is based on the size of the queue.
cols = 128
rows = cols * 2
grid_size = q.qsize() / 2 if q.qsize() % 2 == 0 else math.fabs(q.qsize() / 2) + 1
# Convert to integer value to prevent 'VisibleDeprecationWarning'
grid_size = int(grid_size)
queue_index = 0
top_index = 0
bottom_index = 0
# Create new (big) image that serves as
img_gallery = np.zeros((rows, grid_size * cols, 3), np.uint8)
scaling_factor = 0.5

# Pull images out from priority queue and place them in our image gallery
# from top left to bottom right
while not q.empty():
    (priority, img) = q.get()
    search_result = cv2.imread(img)
    search_result = cv2.resize(search_result, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Find ROI in top row
    if queue_index < grid_size:
        img_gallery[0:128, top_index * cols:top_index * cols + cols] = search_result
        top_index += 1

    # Find ROI in bottom row
    if queue_index >= grid_size:
        img_gallery[128:257, bottom_index * cols:bottom_index * cols + cols] = search_result
        bottom_index += 1

    queue_index += 1


# Display and arrange output
cv2.imshow('Query Input Image', query_img)
cv2.moveWindow('Query Input Image', 0, 0)
cv2.imshow('Image Retrieval Result', img_gallery)
cv2.moveWindow('Image Retrieval Result', 0, 310)

cv2.waitKey(0)
cv2.destroyAllWindows()
