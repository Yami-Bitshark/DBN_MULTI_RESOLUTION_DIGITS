#!/usr/bin/python
#./gen.py img_src label 

import sys
import numpy as np
import cv2
import subprocess


def tup(i):
    """docstring for tup"""
    return (i,int(i*8/5))


arg1=str(sys.argv[1])

img=cv2.imread(arg1)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.equalizeHist(img)
print img.shape
if img.shape != (11,7):
    img=cv2.resize(img,(7,11),interpolation=cv2.INTER_CUBIC)
    print img.shape


for i in range(2,8):
    y=cv2.resize(img,tup(i),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    print y.shape
    y=cv2.resize(y,(7,11),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    cv2.imwrite("exp/"+str(sys.argv[2])+"_"+str(tup(i)[1])+"x"+str(tup(i)[0])+".png",y)

