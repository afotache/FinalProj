import cv2
import numpy as np

# Read image
im = cv2.imread("largebib.png", cv2.IMREAD_GRAYSCALE)
for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > 128:
                im  [i,j] = 255
            else:
                im[i,j] = 0  
              
cv2.imwrite("largebibba.png", im)
cv2.imshow("Keypoints", im)
cv2.waitKey(0)
'''  
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 1500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
     
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
'''

img = cv2.imread('realchris2.png')
gray = cv2.imread('realchris2.png',0)

ret,thresh = cv2.threshold(gray,127,255,1)

img3, contours,h = cv2.findContours(thresh,1,2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print len(approx)
    if len(approx)==5:
        print "pentagon"
        cv2.drawContours(img,[cnt],0,255,-1)
    #elif len(approx)==3:
    #    print "triangle"
    #    cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    elif len(approx)==4:
        print "square"
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    #elif len(approx) == 9:
    #    print "half-circle"
    #    cv2.drawContours(img,[cnt],0,(255,255,0),-1)
    #elif len(approx) > 15:
    #    print "circle"
    #    cv2.drawContours(img,[cnt],0,(0,255,255),-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
##########another version - doesn't work

#cap = cv2.VideoCapture(0)
im = cv2.imread('realchris.png')
#_, frame = cap.read()
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# define range of white color in HSV
# change it according to your need !
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([0,0,255], dtype=np.uint8)

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(im,im, mask= mask)

#cv2.imshow('im',im)
#cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.waitKey(0)
'''