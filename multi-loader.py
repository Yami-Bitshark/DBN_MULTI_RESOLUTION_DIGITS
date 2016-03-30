#!/usr/bin/python





import numpy as np
import os
import cv2
import sys
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
from sklearn.externals import joblib
import matplotlib.pyplot as plt

#ploting the report
def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):
    lines = cr.split('\n')
    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        #print(v)
        plotMat.append(v)
    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show() 


def tup(i):
    """docstring for tup"""
    return int(i*8/5)








################################### training data
def db_load(r,c):
    """loads images from folders and creates a db for a given resolution rowsxcolumns"""
    #label matrix 
    label=[]
    nmb=0
    for i in range(0,10):
        for fich in os.listdir("db/"+str(i)):
            label.append(i)
            nmb=nmb+1
    #data matrix: 10 rows (digits): rows(kxtup(k)) each block is a matrix of the images of the corresponding size and digit
    data_set=np.zeros((1,int(r)*int(c)))

    for i in range(0,10):
        for fich in os.listdir("db/"+str(i)):
            img=cv2.imread("db/"+str(i)+"/"+fich,0)
            img=cv2.equalizeHist(img)
            if img.shape != (11,7):
                img=cv2.resize(img,(7,11),interpolation=cv2.INTER_CUBIC)
            samp=cv2.resize(img,(int(c),int(r)),interpolation=cv2.INTER_CUBIC)
            samp=cv2.equalizeHist(samp)
            samp=samp.flatten()/255.0
            data_set=np.vstack((data_set,samp))
                
    data_set=np.delete(data_set, (0), axis=0)
    label=np.asarray(label).reshape(nmb,1)
    return data_set,label




############################################ test data
def test_load(r,c):
    """loads images from folders and creates a test db for a given resolution rowsxcolumns"""
    #label matrix 
    labeltest=[]
    nmb=0
    for i in range(0,10):
        for fich in os.listdir("test/"+str(i)):
            labeltest.append(i)
            nmb=nmb+1
    #data matrix: 10 rows (digits): rows(kxtup(k)) each block is a matrix of the images of the corresponding size and digit
    test_set=np.zeros((1,int(r)*int(c)))

    for i in range(0,10):
        for fich in os.listdir("test/"+str(i)):
            img=cv2.imread("test/"+str(i)+"/"+fich,0)
            img=cv2.equalizeHist(img)
            if img.shape != (11,7):
                img=cv2.resize(img,(7,11),interpolation=cv2.INTER_CUBIC)
            samp=cv2.resize(img,(int(c),int(r)),interpolation=cv2.INTER_CUBIC)
            samp=cv2.equalizeHist(samp)
            samp=samp.flatten()/255.0
            test_set=np.vstack((test_set,samp))
                
    test_set=np.delete(test_set, (0), axis=0)
    labeltest=np.asarray(labeltest).reshape(nmb,1)
    return test_set,labeltest










####################################

if __name__=='__main__':
    dbn_list=[]
    for i in range(2,8):
        dat,lab=db_load(tup(i),i)
        try:
            dbn = joblib.load("pickles/dbn_"+str(tup(i))+"x"+str(i)+".pkl") 
            dbn_list.append(dbn)
        except:
            dbn = DBN(
                [i*tup(i), 400, 10],
                learn_rates = 0.3,
                learn_rate_decays = 0.9,
                epochs = 50,
                verbose = 1
                )
            dbn.fit(dat,lab)
            dbn_list.append(dbn)
            joblib.dump(dbn,"pickles/dbn_"+str(tup(i))+"x"+str(i)+".pkl")
        finally:
            #print dat.shape
            #print lab.shape
            print dbn_list.__len__()
            print ("trained ! ready to predict!")
            print "training report for {}x{}:".format(tup(i),i)
            tes,labt=test_load(tup(i),i)
            preds=dbn.predict(tes)
            sampleClassificationReport=classification_report(labt,preds)
            print sampleClassificationReport

    while(1):
        dst="."
        dst=str(raw_input("image to test? \'q\' to quit:\n"))
        if dst == "q":
            break
        else:
            try:
                k=0
                for i in range(2,8):

                    img=cv2.imread(dst,0)
                    img=cv2.equalizeHist(img)
                    if img.shape != (tup(i),i):
                        img=cv2.resize(img,(i,tup(i)),interpolation=cv2.INTER_CUBIC)
                    img=img.reshape(1,i*tup(i))/255.0
                    
                    #prediction:
                    pred=dbn_list[k].predict(img)
                    k=k+1
                    print "prediction for ({},{}) : {}".format(tup(i),i,pred)

            except:
                print "error reading image.."




























