import numpy as np
import cv2
import video
import time
import sys
from glob import glob
import itertools as it

'''
try:
        video = cv2.VideoCapture("colin.mov")        
        rval , frame = video.read()
        cv2.imwrite('colinnum.png', frame) 
except:
        print "Could not open video file"
        raise
print video.grab()

'''

vidcap = cv2.VideoCapture("IMG_6233.MOV")  
success,image = vidcap.read()

count = 0;
while success:
    success,image = vidcap.read()
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
    count += 1
    if count == 180:
        cv2.imwrite("frame%d.jpg" % count, image)
print count    