# -*- coding: utf-8 -*-

#this file is first

import scipy
import sys,os,cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

tStart = time.time()
path = os.getcwd()
data_train = pd.read_csv(path + '/' + 'train.csv', low_memory=False)
data_t_l=data_train.values[:,:]
data_dev = pd.read_csv(path + '/' + 'dev.csv', low_memory=False)
data_d_l=data_dev.values[:,:]
name = data_train.values[0:300,:]
b=[]
n = name[18,10]
b.append(n)
n = name[1,5]
b.append(n)
n = name[4,5]
b.append(n)
n = name[2,10]
b.append(n)
n = name[281,15]
b.append(n)

def label(data_t,label_name):
    ndata=[]
    a=np.shape(data_t)
    l=[]
    a1=0
    a2=0
    a3=0
    a4=0
    a5=0
    ndata=data_t[:,0]
    for i in range(0, a[0]):    
        for j in range(5,a[1]):
            if data_t[i,j] == label_name[0]:
                a1=a1+1
            
            if data_t[i,j] == label_name[1]:
                a2=a2+1
            
            if data_t[i,j] == label_name[2]:
                a3=a3+1   
            
            if data_t[i,j] == label_name[3]:
                a4=a4+1
            
            if data_t[i,j] == label_name[4]:
                a5=a5+1 
                
        if a1==0:
            x=0       
        else:
            x=1
        l.append(x)
    
        if a2==0:
            x=0       
        else:
            x=1
        l.append(x)
    
        if a3==0:
            x=0       
        else:
            x=1
        l.append(x)
                
        if a4==0:
            x=0       
        else:
            x=1
        l.append(x)
    
        if a5==0:
            x=0       
        else:
            x=1
        l.append(x)
        
        a1=0
        a2=0
        a3=0
        a4=0
        a5=0         
            
    c=np.reshape(l, (a[0], 5))
    t_l=np.reshape(ndata, (len(ndata),1))
    pdata = np.hstack((t_l,c))
    return pdata

train_label_data = label(data_t_l,b)
dev_label_data = label(data_d_l,b)
                   

with open("train_c_label.csv","w",encoding='utf-8') as file:
        file.write('image_id,'+str(b[0])+','+str(b[1])+','+str(b[2])+','+str(b[3])+','+str(b[4])+'\n')
        for i in range(len(train_label_data)):
            file.write(str(train_label_data[i][0]) + ',' + str(train_label_data[i][1]) + ',' + str(train_label_data[i][2])+ ',' + str(train_label_data[i][3])+ ',' + str(train_label_data[i][4])+ ',' + str(train_label_data[i][5])+ '\n')
            
with open("dev_c_label.csv","w",encoding='utf-8') as file:
        file.write('image_id,'+str(b[0])+','+str(b[1])+','+str(b[2])+','+str(b[3])+','+str(b[4])+'\n')
        for i in range(len(dev_label_data)):
            file.write(str(dev_label_data[i][0]) + ',' + str(dev_label_data[i][1]) + ',' + str(dev_label_data[i][2])+ ',' + str(dev_label_data[i][3])+ ',' + str(dev_label_data[i][4])+ ',' + str(dev_label_data[i][5])+ '\n')

                  
tEnd = time.time()    
print("It cost %f sec",(tEnd - tStart))    