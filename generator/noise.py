#!/usr/bin/python

import numpy as np
import os
import cv2
import sys
from scipy import ndimage

def noisy(noise_typ,image):

    if noise_typ == "gausblur":
        noisy = ndimage.gaussian_filter(image, sigma=2)
        return noisy

    elif noise_typ == "salt":
        row,col = image.shape
        s_vs_p = 0.5 
        amount = 0.15
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "pois":
        PEAK= np.random.randn()
        if PEAK < 0.1:
            PEAK=0.1
        nois = np.random.poisson(image / 255.0 * PEAK) / PEAK * 255.0  # noisy image
        noisy= (image + nois)
        dat=image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                dat[i][j]=noisy[i][j]
        return dat

    else:
        print "Error Using function"




for i in range(0,10):
    folder=str(i)+"/"
    for fld in os.listdir(folder):
        path=folder+"/"+fld
        cnt=os.listdir(path)
        for image in cnt:
            namein=path+"/"+image
            img=cv2.imread(namein,0)
            for j in ["gausblur","salt","pois"]:
                nameout=path+"/"+j+"_"+image
                cv2.imwrite(nameout,noisy(j,img))
            os.remove(namein)




