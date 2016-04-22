#!/usr/bin/python
import os
import sys
import numpy as np
import cv2
from scipy import ndimage

def noisy(image,siz,namein):

    namein=str(namein) 
    print namein
    n1 = ndimage.gaussian_filter(image, sigma=1)
        
    cv2.imwrite("gaus_1_"+namein,n1)
    n2 = ndimage.gaussian_filter(image,sigma=3)
    cv2.imwrite("gaus_3_"+namein,n2)
    if siz > 25:
        n3 = n2.gaussian_filter(image,sigma=5)
        cv2.imwrite("gaus_5_"+namein,n3)
    if siz > 49:
        n4 = ndimage.gaussian_filter(image,sigma=7)
        cv2.imwrite("gaus_7_"+namein,n4)
    if siz > 81:
        n5 = ndimage.gaussian_filter(image,sigma=9)
        cv2.imwrite("gaus_9_"+namein,n5)

    k=0
    for h in [0.05,0.1,0.15,0.2,0.3]:
         row,col = image.shape
         s_vs_p = 0.5 
         amount = h
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
         cv2.imwrite("salt_"+str(k)+"_"+namein,out)
         k+=1
    k=0
    for h in range(6):
        PEAK= np.random.randn()
        if PEAK < 0.1:
            PEAK=0.1
        nois = np.random.poisson(image / 255.0 * PEAK) / PEAK * 255.0  # noisy image
        noisy= (image + nois)
        dat=image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                dat[i][j]=noisy[i][j]
        cv2.imwrite("poiss_"+str(k)+"_"+namein,dat)
        k+=1


def tup1(i):
    """docstring for tup"""
    return (i,int(i*8/5))



#####################################################################


db="test-seq-noise"

for i in range(0,10):
    os.mkdir("{}/{}/3x2".format(db,i))
    os.mkdir("{}/{}/4x3".format(db,i))
    os.mkdir("{}/{}/6x4".format(db,i))
    os.mkdir("{}/{}/8x5".format(db,i))
    os.mkdir("{}/{}/9x6".format(db,i))
    os.mkdir("{}/{}/11x7".format(db,i))

for fl in [ "{}/{}".format(db,i) for i in range(0,10) ]:
    k=0
    for fc in os.listdir(fl):
        fcc=fl+"/"+fc
        if os.path.isfile(fcc):
            img=cv2.imread(fcc,0)
            img=cv2.equalizeHist(img)
            for i in range(2,8):
                res=str(tup1(i)[1])+"x"+str(tup1(i)[0])
                y=cv2.resize(img,tup1(i),interpolation=cv2.INTER_CUBIC)
                y=cv2.equalizeHist(y)
                cv2.imwrite(fc+"/"+res+"/"+str(k)+"_"+res+".png",y)
            os.remove(fcc)
    k+=1


  






#namein=str(sys.argv[1])
#lab=lab+".png"
#img=cv2.imread(namein,0)
#siz = img.shape[0]*img.shape[1]
#noisy(img,siz,namein)







