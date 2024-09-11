# -*- coding: utf-8 -*-

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

lower = np.array([30,80,0])
upper = np.array([64,359,359])

_,background = cam.read()

kernel= np.ones((3,3),np.uint8)
kernel2=np.ones((11,11),np.uint8)
kernel3=np.ones((13,13),np.uint8)

while(cam.isOpened()):
    _, frame = cam.read()
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    
    mask = cv2.inRange(hsv,lower,upper)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel3, iterations=2)

    mask_not = cv2.bitwise_not(mask)
    
    bg = cv2.bitwise_and(background, background, mask = mask)
    fg = cv2.bitwise_and(frame, frame, mask = mask_not)

    dst = cv2.addWeighted(bg, 1, fg, 1, 0)
    

    cv2.imshow("original",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("dst",dst)
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    

cv2.destroyAllWindows()