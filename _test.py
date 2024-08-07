"""
Date: 2024/07/23
@author: OAO
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from queue import Queue
from threading import Thread
import warnings
import numpy as np
import h5py as h5
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

print('-'*30)
model=load_model('HiLo220511_fromYH.h5', compile=False) # Load the model
PATH='D:/240722_YuHsinModel/test_Data' # Load all the figures
# image stack size (width, hieght, number of images)
IMG_WIDTH = 256 #477
IMG_HEIGHT = 256 #588
IMG_NUM = 6


def trained_dataset():
    """
    Read the file and store into the images
    """
    dirs = os.listdir(PATH)
    images = np.zeros((IMG_NUM,IMG_HEIGHT,IMG_WIDTH))  #Set the 3d empty matrix    
    for index, filename in dirs:   #用for迴圈把圖逐張放到tensor當中
        img=cv.imread(PATH+"/"+filename, cv.IMREAD_GRAYSCALE)
        img=tf.cast(img, dtype=tf.dtypes.float32)
        img=tf.abs(tf.pow(img,1))
        img=img/tf.pow(255., 1)
        #img=img/np.amax(img)
        img= np.array(img)
        img= cv.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv.INTER_AREA)
        images [:,:,index] = img
        i=i+1

    imgs_train=img3d.reshape((IMG_NUM, IMG_WIDTH, IMG_HEIGHT, 1))  #tensor要記得reshape才能丟進去model中  
    return img3d


img3d = trained_dataset()


#test的重建

for i in range (0,IMG_NUM):
    train=img3d[:,:,i]
    train=train.reshape((1, 256, 256, 1))
    predtrain=model.predict(train)
    predtrainimg=predtrain.reshape((256, 256))
    plt.figure(1)
    plt.imshow(predtrainimg,cmap='gray')
    predtrainimg1=predtrainimg*255
    predtrainimg1=predtrainimg1.astype(np.uint8)
    filename='HiLopred_%i.tiff'%i
    cv.imwrite(filename,predtrainimg1)
         


#plt.imshow(tstphasefinal,cmap='gray')
#cv.imwrite('15phase.tiff',tstphasefinal)
#img=255*predimgfinal
#img=tf.abs(tf.pow(predimgfinal, 2))
#img=img.astype(np.uint8)

# plt.figure(1)
# plt.imshow(predtrainimg,cmap='gray')
#plt.figure(2)
#plt.imshow(predtrainimg,cmap='jet')

# predtrainimg1=predtrainimg*255
# predtrainimg1=predtrainimg1.astype(np.uint8)
# cv.imwrite('HiLopred.tiff',predtrainimg1)
#cv.imwrite('test1953phase.tiff',img)
