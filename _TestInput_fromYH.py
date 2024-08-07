# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:34:55 2021

@author: s9314
"""

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
import os
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from DeepCGHtest_8_Fraunhofer_Random_img_2 import __unet, traindataset, interleave, deinterleave, __phi_slm, __prop__, __accuracy, __ifft_AmPh
#from HiLoDLtest_2_allimage_dropout_5 import __unet, traindataset

#model=load_model('DeepCGH_simple_2.h5', custom_objects={'Interleave': interleave,'phi_0': deinterleave, 'amp_0':deinterleave, 'phi_slm':__ifft_AmPh, 'phi_slm_':__phi_slm, 'pred_img':__prop__,'loss':__accuracy })
#model=load_model('DeepCGH_test6_2.h5',custom_objects={'Interleave': interleave,'phi_0': deinterleave, 'amp_0':deinterleave, 'phi_slm_':__phi_slm, 'pred_img':__prop__},compile=False)
model=load_model('HiLo220511_1_Unet_ResNet_MAE_batch14_Moredata3_UpSampling_bilinear_k3_drop_allLReLU(0.5)_big_BNmore.h5', compile=False)


#path='C:/Users/WS1030/Desktop/Yu-Hsin/HiLo_DL/Test_Input'
#path='C:/Users/WS1030/Desktop/Yu-Hsin/HiLo_DL/Valtest'
#path='C:/Users/WS1030/Desktop/Yu-Hsin/HiLo_DL/Time_test'
path='C:/Users/WS1030/Desktop/Yu-Hsin/HiLo_DL/22_09_13 In vivo mice brain_AdjustBrightness(0.7)'
#path='C:/Users/WS1030/Desktop/Yu-Hsin/HiLo_DL/Mean_absolute_map/Train/Input'
#path2='C:/Users/s9314/Desktop/Yu_Hsin/HiLo_DL/HiLo_cut/Test'
#path3='C:/Users/s9314/Desktop/Yu_Hsin/HiLo_DL/Uni_cut/Train'
#path4='C:/Users/s9314/Desktop/Yu_Hsin/HiLo_DL/Uni_cut/Test'


def traindataset():
    dirs = os.listdir(path)
    i=0
    img3d = np.zeros((256,256,103))  #設一個3d 256x256x5520的空矩陣 for training input
    for filename in dirs:   #用for迴圈把圖逐張放到tensor當中
        img=cv.imread(path+"/"+filename, cv.IMREAD_GRAYSCALE)
        img=tf.cast(img, dtype=tf.dtypes.float32)
        img=tf.abs(tf.pow(img,1))
        img=img/tf.pow(255., 1)
        #img=img/np.amax(img)
        img= np.array(img)
        img= cv.resize(img, (256, 256), interpolation=cv.INTER_AREA)
        img3d [:,:,i] = img
        i=i+1
        
    imgs_train=img3d.reshape((103, 256, 256, 1))  #tensor要記得reshape才能丟進去model中  
    return img3d


img3d = traindataset()


#test的重建

for i in range (0,82):
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

plt.figure(1)
plt.imshow(predtrainimg,cmap='gray')
#plt.figure(2)
#plt.imshow(predtrainimg,cmap='jet')

predtrainimg1=predtrainimg*255
predtrainimg1=predtrainimg1.astype(np.uint8)
cv.imwrite('HiLopred.tiff',predtrainimg1)
#cv.imwrite('test1953phase.tiff',img)
