# -*- coding: utf-8 -*-

#this file is third
import scipy
from keras.layers import Dense
import sys,os,cv2
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras import layers
from keras.models import load_model
import time
tStart = time.time()

path = os.getcwd()
model = load_model(path + '/' + 'mango.h5')

test_data_name = pd.read_csv(path + '/' + 'Test_UploadSheet.csv')
img_name = test_data_name['image_id'].values

test_img_list = []
for img in img_name: 
    input_img=cv2.imread(path + '/Test/' + img)
    input_img_resize=cv2.resize(input_img, (160,120))
    test_img_list.append(input_img_resize)
    print('progress:' + img)
    
test_img_data = np.array(test_img_list)
test_img_data = test_img_data.astype('float64')
test_img_data /= 255

y_predict = model.predict(test_img_data, batch_size=128)

oa0 = []
oa1 = []
oa2 = []
oa3 = []
oa4 = []
for i in range(len(y_predict)):
    np0 = y_predict[i][0]
    if np0 > 0.1:
        np0 = 1
    else:
        np0 = 0
    np1 = y_predict[i][1]
    if np1 > 0.1:
        np1 = 1
    else:
        np1 = 0
    np2 = y_predict[i][2]
    if np2 > 0.1:
        np2 = 1
    else:
        np2 = 0
    np3 = y_predict[i][3]
    if np3 > 0.1:
        np3 = 1
    else:
        np3 = 0
    np4 = y_predict[i][4]
    if np4 > 0.1:
        np4 = 1
    else:
        np4 = 0
    oa0 += [np0]
    oa1 += [np1]
    oa2 += [np2]
    oa3 += [np3]
    oa4 += [np4]

with open("predit.csv","w",encoding='utf-8') as file:
        file.write("image_id,D1,D2,D3,D4,D5\n")
        for i in range(len(y_predict)):
            file.write(str(img_name[i]) + ',' + str(oa0[i]) + ',' + str(oa1[i])+ ',' + str(oa2[i])+ ',' + str(oa3[i])+ ',' + str(oa4[i])+ '\n')

tEnd = time.time()    
print("It cost %f sec",(tEnd - tStart))  