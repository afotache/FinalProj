#im = cv2.imread('realbibagain.png')  #this one worked also
#im = cv2.imread('59.jpg')  #this one worked also
#im = cv2.imread('googlebib.png')    #this one worked also

# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
from nolearn.dbn import DBN
import numpy as np
import cv2

dbn = joblib.load("digitsdbn.pkl")

#im = cv2.imread('n3.jpg') #this worked
im = cv2.imread('IMG_6237.JPG') 
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
        cv2.rectangle(im,(x+facex-int(facew/4),y+facey+faceh*3),(x+facex-int(facew/4)+w,y+facey+faceh*3+h),(0,0,255),1)
        roi = im_th[y-5:y+h+10, x-5:x+w+10]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #roi = cv2.dilate(roi, (3, 3))
        #cv2.erode(roi, None, iterations=2)
        xroi = np.array(roi)
        digit = xroi.reshape(-1,784).astype(np.float32)
        pred = dbn.predict(digit)
        print pred
        cv2.putText(im, str(int(pred[0])), (x+facex-int(facew/4),y+facey+faceh*3),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)


cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

