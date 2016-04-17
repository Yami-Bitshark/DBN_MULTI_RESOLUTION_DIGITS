#!/usr/bin/python
#./gen.py img_src label 

import sys
import numpy as np
import cv2
import subprocess


def tup1(i):
    """docstring for tup"""
    return (i,int(i*8/5))
def tup2(i):
    """docstring for tup"""
    return (i,int(i*8/6))
def tup3(i):
    """docstring for tup"""
    return (i,int(i*8/4))
def tup4(i):
    """docstring for tup"""
    return (i,int(i*8/3))
def tup5(i):
    """docstring for tup"""
    return (i,i)


arg1=str(sys.argv[1])

img=cv2.imread(arg1)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.equalizeHist(img)
print img.shape
if img.shape != (11,7):
    img=cv2.resize(img,(7,11),interpolation=cv2.INTER_CUBIC)
    


for i in range(2,11):
    print "{}x{}".format(tup1(i),i)
    y=cv2.resize(img,tup1(i),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    cv2.imwrite("exp2/"+str(sys.argv[2])+"_"+str(tup1(i)[1])+"x"+str(tup1(i)[0])+".png",y)
    print "{}x{}".format(tup2(i),i)
    y=cv2.resize(img,tup2(i),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    cv2.imwrite("exp2/"+str(sys.argv[2])+"_"+str(tup2(i)[1])+"x"+str(tup2(i)[0])+".png",y)
    print "{}x{}".format(tup3(i),i)
    y=cv2.resize(img,tup3(i),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    cv2.imwrite("exp2/"+str(sys.argv[2])+"_"+str(tup3(i)[1])+"x"+str(tup3(i)[0])+".png",y)
    print "{}x{}".format(tup4(i),i)
    y=cv2.resize(img,tup4(i),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    cv2.imwrite("exp2/"+str(sys.argv[2])+"_"+str(tup4(i)[1])+"x"+str(tup4(i)[0])+".png",y)
    print "{}x{}".format(tup5(i),i)
    y=cv2.resize(img,tup5(i),interpolation=cv2.INTER_CUBIC)
    y=cv2.equalizeHist(y)
    cv2.imwrite("exp2/"+str(sys.argv[2])+"_"+str(tup5(i)[1])+"x"+str(tup5(i)[0])+".png",y)










