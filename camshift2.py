#!/usr/bin/env python

import numpy as np
import cv2
import video
import time
import sys
from glob import glob
import itertools as it
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
from nolearn.dbn import DBN


class App(object):
    def __init__(self):
        self.cam=cv2.VideoCapture('IMG_6233.MOV')
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.persondetected=0
        self.show_backproj = False
        self.sizex=0
        self.sizey=0
        self.imgwidth=960
        self.imgheight=540

    def inside(self, r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def draw_detections(self, img, rects, thickness = 1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w = int(0.2*w)
            pad_h = int(0.2*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
            self.selection = (x+pad_w, y+pad_h, x+w-pad_w, y+h-pad_h)
            self.sizex=w
            self.sizey=h
            self.tracking_state = 1

    def run(self):
        while True:
            ret, self.frame = self.cam.read()            
            self.frame = cv2.resize(self.frame, (self.imgwidth, self.imgheight))
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                self.hist = hist.reshape(-1)
                #self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.tracking_state == 1:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                #print track_box
                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try:
                    #pts = cv2.cv.BoxPoints(track_box)
                    #pts = np.int0(pts)
                    #cv2.polylines(vis, [pts],True, (0, 0, 255),2)
                    #cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                    pass
#                     x1=int(track_box[0][0]-track_box[1][0]/2)
#                     y1=int(track_box[0][1]-track_box[1][1]/2)
#                     x2=int(track_box[0][0]+track_box[1][0]/2)
#                     y2=int(track_box[0][1]+track_box[1][1]/2)
#                     cv2.rectangle(vis, (x1,y1), (x2,y2), (0, 255, 0), 2)
#                     cv2.line(vis, (x1-20,int((y2+y1)/2)), (x1,int((y2+y1)/2)), (0, 255, 0), 2)
#                     cv2.line(vis, (x2,int((y2+y1)/2)), (x2+20,int((y2+y1)/2)), (0, 255, 0), 2)
#                     cv2.line(vis, (int((x2+x1)/2), y1-20), (int((x2+x1)/2), y1), (0, 255, 0), 2)
#                     cv2.line(vis, (int((x2+x1)/2), y2), (int((x2+x1)/2), y2+20), (0, 255, 0), 2)
#                     cv2.putText(vis,"FRIENDLY", (x1-15,y2+40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))                    
                except OSError as err:
                    #print track_box
                    print OSError
                #print track_box
                #print track_box[1][0]
                if (track_box[1][0] / self.sizex > 1.2) & (track_box[1][1] / self.sizey > 1.2):
                    self.tracking_state=0
                    self.persondetected=0

            #cv2.imshow('camshift', vis)
            if self.persondetected==0:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
                img=vis
                found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
                found_filtered = []
                for ri, r in enumerate(found):
                    for qi, q in enumerate(found):
                        if (ri != qi and self.inside(r, q)) or r[0]==0:
                            break
                        else:
                            found_filtered.append(r)
                #self.draw_detections(img, found)
                self.draw_detections(img, found_filtered, 3)
                cv2.imshow('camshift', img)
                if self.sizey > self.imgheight*.7:
                    cv2.imwrite('detected.png', img)
                    
                    cv2.waitKey(0)
                    break
                    
                if len(found_filtered) > 0:
                    #print len(found_filtered), len(found), found_filtered[0][0]                    
                    #cv2.imwrite('coco.png', img)
                    #cv2.waitKey(0)
                    self.persondetected=1
    
            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print __doc__
    App().run()
    #img3 = cv2.imread('coco.png') 
    #cv2.imshow('chris', img3)
    #cv2.waitKey(0)
    dbn = joblib.load("digitsdbn.pkl")
    #im = cv2.imread('n3.jpg') #this worked
    im = cv2.imread('detected2.png') 
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    faceh=faces[0][2]
    facew=faces[0][3]
    facex=faces[0][0]
    facey=faces[0][1]
    (x,y,w,h) = faces[0]
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
    bibroi=im[y+h*3:y+h*5, x-int(w/4):x+int(w*1.5)]
    print w,h
    bibgray = cv2.cvtColor(bibroi,cv2.COLOR_BGR2GRAY)
    ret, im_th = cv2.threshold(bibgray, 90, 255, cv2.THRESH_BINARY_INV)
    #im_th = cv2.erode(im_th, None, iterations=2)
    cv2.imshow("Resulting Image with Rectangular ROIs", im_th)
    cv2.waitKey(0)
    
    imcont, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for pcont in ctrs:
        [x,y,w,h] = cv2.boundingRect(pcont)
        if h<faceh*.7 and h>faceh*.3 and h/w>1.1:
            #cv2.rectangle(im,(x+facex-int(facew/4),y+facey+faceh*3),(x+facex-int(facew/4)+w,y+facey+faceh*3+h),(0,0,255),1)
            roi = im_th[y-5:y+h+10, x-5:x+w+10]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            #cv2.erode(roi, None, iterations=2)
            xroi = np.array(roi)
            digit = xroi.reshape(-1,784).astype(np.float32)
            pred = dbn.predict(digit)
            cv2.putText(im, str(int(pred[0])), (x+facex-int(facew/4),y+facey+faceh*3),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(0)