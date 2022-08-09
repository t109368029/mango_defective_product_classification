# -*- coding: utf-8 -*-

#this file is second
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
from keras.callbacks import EarlyStopping
from keras import metrics
from keras import layers
import time
tStart = time.time()

path = os.getcwd()

train_data = pd.read_csv(path + '/' + 'train_c_label.csv')
pd.DataFrame(train_data)

valid_data = pd.read_csv(path + '/' + 'dev_c_label.csv')
pd.DataFrame(valid_data)

img_train_name = train_data['image_id'].values
train_label = train_data.iloc[:,1:].values
train_label.astype(np.float64)


img_valid_name = valid_data['image_id'].values
valid_label = valid_data.iloc[:,1:].values
valid_label.astype(np.float64)

img_data_list=[]
for img in img_train_name: 
    input_img=cv2.imread(path + '/Train/' + img)
    input_img_resize=cv2.resize(input_img, (160,120),interpolation=cv2.INTER_LINEAR)
    img_data_list.append(input_img_resize)
    print('progress:' + img)
    
valid_img_list=[]
for img in img_valid_name: 
    input_img=cv2.imread(path + '/Dev/' + img)
    input_img_resize=cv2.resize(input_img,(160, 120))
    valid_img_list.append(input_img_resize)
    print('progress:' + img)

train_img_data = np.array(img_data_list)
train_img_data = train_img_data.astype('float64')
train_img_data /= 255

valid_img_data = np.array(valid_img_list)
valid_img_data = valid_img_data.astype('float64')
valid_img_data /= 255

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 160, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(300, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size=128
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
history = model.fit(train_img_data, train_label,batch_size=batch_size, epochs=300, validation_data=(valid_img_data, valid_label), callbacks=[early_stopping])
model.save(path + '/' + 'mango.h5')

plt.plot(range(1, len(history.history['loss']) + 1), history.history['acc'])
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.show()

plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()

tEnd = time.time()    
print("It cost %f sec",(tEnd - tStart))  