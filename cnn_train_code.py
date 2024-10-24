# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 22:21:38 2024

@author: PZC
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:16:04 2024

@author: john
"""

import os
import math
import numpy as np
import tensorflow
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Flatten,concatenate,Input,Conv1D,MaxPooling1D,GlobalAveragePooling1D,Dense,Dropout
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt

#设置学习率随epoch下降
def step_decay(epoch):
    ini_rate,drop=0.0001,0.5
    epochs_drop=200
    rate=ini_rate*math.pow(drop,math.floor((1+epoch)/epochs_drop)) #每100个epoch下降一半的学习率
    #当rate递减过5次以后，将不再降低
    if epoch>=1000:
        rate=ini_rate*math.pow(drop,1000/epochs_drop)
    return rate
#计算tss
def cal_tss(Y_train):
    mean=np.mean(Y_train)
    Y_train=Y_train-mean
    result=np.sum(np.square(Y_train))
    return result
#cnn_model模块
def cnn_model(X_input):
    X_input=Input(X_input.shape[1:],name='input')
    #(1)普通卷积
    X=Conv1D(32,1,strides=1,padding='same',name='conv1')(X_input)
    #(2)Inception模块-1
    #(2.1)支线1
    th0=Conv1D(32,1,strides=1,padding='same',activation='relu',name='incep0_1')(X)
    #(2.2)支线2
    th1=Conv1D(32,1,strides=1,padding='same',activation='relu',name='incep1_1')(X)
    th1=Conv1D(32,3,strides=1,padding='same', activation='relu',name='incep1_2')(th1)
    #(2.3)支线3
    th2=Conv1D(32,1,strides=1,padding='same',activation='relu',name='incep2_1')(X)
    th2=Conv1D(32,5,strides=1,padding='same', activation='relu',name='incep2_2')(th2)
    #(2.4)支线4
    th3=MaxPooling1D(3,strides=1,padding='same',name='incep3_1')(X)
    th3=Conv1D(32,1,strides=1,padding='same', activation='relu',name='incep3_2')(th3)    
    #(2.4)将支线合并,并加入residual模块
    X=concatenate([th0,th1,th2,th3], axis=1,name='concat')
    #(3)将三维数据转化为二维
    X=GlobalAveragePooling1D(data_format='channels_first',name='ga_pool')(X)
    #(4)建立各指标独立的Dense模块
    #(4.1)灵芝酸C2模块
    fc1=Dense(256,activation='relu',name='fc1_1')(X)
    fc1=Dropout(0.1,name='dropout1_1')(fc1)
    fc1=Dense(128,activation='relu',name='fc1_2')(fc1)
    fc1=Dropout(0.1,name='dropout1_2')(fc1)
    fc1=Dense(32,activation='relu',name='fc1_3')(fc1)
    fc1=Dense(1,name='LZSC2')(fc1)
    #(4.2)灵芝酸C6模块
    fc2=Dense(256,activation='relu',name='fc2_1')(X)
    fc2=Dropout(0.1,name='dropout2_1')(fc2)
    fc2=Dense(128,activation='relu',name='fc2_2')(fc2)
    fc2=Dropout(0.1,name='dropout2_2')(fc2)
    fc2=Dense(32,activation='relu',name='fc2_3')(fc2)
    fc2=Dense(1,name='LZSC6')(fc2)
    #(4.3)灵芝酸G模块
    fc3=Dense(256,activation='relu',name='fc3_1')(X)
    fc3=Dropout(0.1,name='dropout3_1')(fc3)
    fc3=Dense(128,activation='relu',name='fc3_2')(fc3)
    fc3=Dropout(0.1,name='dropout3_2')(fc3)
    fc3=Dense(32,activation='relu',name='fc3_3')(fc3)
    fc3=Dense(1,name='LZSG')(fc3)
    #(4.4)灵芝酸B模块
    fc4=Dense(256,activation='relu',name='fc4_1')(X)
    fc4=Dropout(0.1,name='dropout4_1')(fc4)
    fc4=Dense(128,activation='relu',name='fc4_2')(fc4)
    fc4=Dropout(0.1,name='dropout4_2')(fc4)
    fc4=Dense(32,activation='relu',name='fc4_3')(fc4)
    fc4=Dense(1,name='LZSB')(fc4)
    #(4.5)灵芝酸A模块
    fc5=Dense(256,activation='relu',name='fc5_1')(X)
    fc5=Dropout(0.1,name='dropout5_1')(fc5)
    fc5=Dense(128,activation='relu',name='fc5_2')(fc5)
    fc5=Dropout(0.1,name='dropout5_2')(fc5)
    fc5=Dense(32,activation='relu',name='fc5_3')(fc5)
    fc5=Dense(1,name='LZSA')(fc5)
    #(4.6)灵芝酸D2模块
    fc6=Dense(256,activation='relu',name='fc6_1')(X)
    fc6=Dropout(0.1,name='dropout6_1')(fc6)
    fc6=Dense(128,activation='relu',name='fc6_2')(fc6)
    fc6=Dropout(0.1,name='dropout6_2')(fc6)
    fc6=Dense(32,activation='relu',name='fc6_3')(fc6)
    fc6=Dense(1,name='LZSD2')(fc6)
    #(4.7)赤芝酸A模块
    fc7=Dense(256,activation='relu',name='fc7_1')(X)
    fc7=Dropout(0.1,name='dropout7_1')(fc7)
    fc7=Dense(128,activation='relu',name='fc7_2')(fc7)
    fc7=Dropout(0.1,name='dropout7_2')(fc7)
    fc7=Dense(32,activation='relu',name='fc7_3')(fc7)
    fc7=Dense(1,name='CZSA')(fc7)
    #(4.8)灵芝烯酸D模块
    fc8=Dense(256,activation='relu',name='fc8_1')(X)
    fc8=Dropout(0.1,name='dropout8_1')(fc8)
    fc8=Dense(128,activation='relu',name='fc8_2')(fc8)
    fc8=Dropout(0.1,name='dropout8_2')(fc8)
    fc8=Dense(32,activation='relu',name='fc8_3')(fc8)
    fc8=Dense(1,name='LZXSD')(fc8)
    #(4.9)灵芝酸C1模块
    fc9=Dense(256,activation='relu',name='fc9_1')(X)
    fc9=Dropout(0.1,name='dropout9_1')(fc9)
    fc9=Dense(128,activation='relu',name='fc9_2')(fc9)
    fc9=Dropout(0.1,name='dropout9_2')(fc9)
    fc9=Dense(32,activation='relu',name='fc9_3')(fc9)
    fc9=Dense(1,name='LZSC1')(fc9)
    #(5)返回模型
    model=Model(inputs=X_input,outputs=[fc1,fc2,fc3,fc4,fc5,fc6,fc7,fc8,fc9],name='cnn_model')
    return model
#cnn_train模块(由于运行效率问题,只运行单折的结果)
def cnn_train(cali_X,cali_Y,rootdir):
    index,sta=0,np.zeros((3,6)) #3:index记录折数;6:sta统计loss,rmse和r2
    #进行五折交叉验证
    kfold=KFold(n_splits=3) #做3折交叉验证
    for train,val in kfold.split(cali_X): #五折交叉验证划分
        print('第'+str(index)+'折神经网络训练')
        #划分训练集和内部验证集
        X_train,X_val=cali_X[train],cali_X[val]
        Y_train,Y_val=cali_Y[train],cali_Y[val]
        #开始神经网络训练
        model=cnn_model(X_train)
        #定义优化器,loss以及保存路径
        model.compile(optimizer=adam_v2.Adam(lr=0.0001),loss=['mse','mse','mse','mse','mse','mse','mse','mse','mse']) #仅输出loss就足够,无需设置metrics
        #定义四个不同功能的callbacks函数
        checkpointer=ModelCheckpoint(filepath=rootdir+'/model_'+str(index)+'.h5',monitor='val_loss',save_best_only=True) #模型的保存路径
        earlystop=EarlyStopping(monitor='val_loss',patience=500) #容忍50epoch结果没有提升
        plateau=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=10,cooldown=10) #防止值陷入高原
        step=LearningRateScheduler(step_decay)
        #tensor_board=TensorBoard(log_dir=rootdir2+'/tmp/log',histogram_freq=0,write_graph=True,write_images=True)
        callbacks=[checkpointer,earlystop,plateau,step]
        #拟合模型
        history=model.fit(X_train,[Y_train[:,0],Y_train[:,1],Y_train[:,2],Y_train[:,3],Y_train[:,4]],epochs=5000,batch_size=50,verbose=2,
                validation_data=(X_val,[Y_val[:,0],Y_val[:,1],Y_val[:,2],Y_val[:,3],Y_val[:,4]]),callbacks=callbacks)
        #输出训练集和测试集的loss
        train_loss=history.history['loss']
        val_loss=history.history['val_loss']
        #保存train_loss,test_loss
        train_loss,val_loss=np.array(train_loss),np.array(val_loss)
        np.save(rootdir+'/train_loss_'+str(index),train_loss)
        np.save(rootdir+'/test_loss_'+str(index),val_loss)
        #绘制loss曲线
        axis_epoch=np.arange(0,len(train_loss))
        plt.figure(figsize=(8,6))
        plt.plot(axis_epoch,train_loss,'g',lw=2)
        plt.plot(axis_epoch,val_loss,'r',lw=2)
        plt.xlabel('epoch')
        plt.legend(['train_loss','val_loss'],loc='upper left')
        plt.savefig(rootdir+'/loss_'+str(index)+'.jpg')
        ##########################
        #以下内容作为参考:train统计数据不准确,test统计数据准确
        #根据训练集和测试集的loss求rmse和r2
        rmse_c,rmse_v=np.sqrt(train_loss),np.sqrt(val_loss)
        rss_c,rss_v=[x*len(X_train) for x in train_loss],[x*len(X_val) for x in val_loss]
        tss_c,tss_v=cal_tss(Y_train),cal_tss(Y_val)
        r2_c,r2_v=1-rss_c/tss_c,1-rss_v/tss_v
        r2_c,r2_v=np.array(r2_c),np.array(r2_v)
        #将数据存入sta中
        arg=np.argmin(val_loss)
        temp_sta=[train_loss[arg],rmse_c[arg],r2_c[arg],val_loss[arg],rmse_v[arg],r2_v[arg]] 
        #绘制acc曲线
        plt.figure(figsize=(8,6))
        plt.plot(axis_epoch,r2_c,'g',lw=2)
        plt.plot(axis_epoch,r2_v,'r',lw=2)
        plt.xlabel('epoch')
        plt.legend(['train_r2','val_r2'],loc='upper left')
        plt.savefig(os.path.join(rootdir,'r2_'+str(index)+'.jpg'))
        plt.show()  
        #将每折的统计结果存入sta
        sta[index]=temp_sta
        index+=1
    #返回temp_sta
    return sta
from sklearn.model_selection import train_test_split
#数据集划分
#随机划分
def random(data, label, test_ratio=0.2, random_state=123):
    """
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :param random_state: the randomseed, default: 123
    :return: X_train :(n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=random_state)

    return X_train, X_test, y_train, y_test
#SPXY划分
def spxy(data, label, test_size=0.2):
    """
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :return: X_train :(n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """
    x_backup = data
    y_backup = label
    M = data.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    label = (label - np.mean(label)) / np.std(label)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = data[i, :]
        ya = label[i]
        for j in range((i + 1), M):
            xb = data[j, :]
            yb = label[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(data.shape[0]), m)

    X_train = data[m, :]
    y_train = y_backup[m]
    X_test = data[m_complement, :]
    y_test = y_backup[m_complement]

    return X_train, X_test, y_train, y_test
#kennard-stone划分
def ks(data, label, test_size=0.2):
    """
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :return: X_train: (n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """
    M = data.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    D = np.zeros((M, M))

    for i in range((M - 1)):
        xa = data[i, :]
        for j in range((i + 1), M):
            xb = data[j, :]
            D[i, j] = np.linalg.norm(xa - xb)

    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    index_column = np.argmax(maxD)

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros((M - i))
        for j in range((M - i)):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(data.shape[0]), m)

    X_train = data[m, :]
    y_train = label[m]
    X_test = data[m_complement, :]
    y_test = label[m_complement]

    return X_train, X_test, y_train, y_test
#划分算法选择
def SetSplit(method, data, label, test_size=0.2, randomseed=123):

    """
    :param method: the method to split trainset and testset, include: random, kennard-stone(ks), spxy
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :return: X_train: (n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """

    if method == "random":
        X_train, X_test, y_train, y_test = random(data, label, test_size, randomseed)
    elif method == "spxy":
        X_train, X_test, y_train, y_test = spxy(data, label, test_size)
    elif method == "ks":
        X_train, X_test, y_train, y_test = ks(data, label, test_size)
    else:
        print("no this  method of split dataset! ")

    return X_train, X_test, y_train, y_test



###模型训练
mean_l = np.loadtxt(r'C:\Users\XX',delimiter = ',')
label_l = np.loadtxt(r'C:\Users\XX',delimiter = ',')
print(mean_l.shape,label_l.shape)
X=(mean_l-np.mean(mean_l,axis=0))/np.std(mean_l,axis=0) #必须对X做归一化,否则结果将会大幅下降
Y=(label_l-np.mean(label_l,axis=0))/np.std(label_l,axis=0) #必须对Y做归一化,否则结果将大幅下降

X=X.reshape(X.shape[0],X.shape[1],1)
cali_X, test_X, cali_Y, test_Y = SetSplit('spxy', X,Y, test_size=0.3, randomseed=123)

#(*4)将数据投入CNN中进行运算
import time
T1=time.time()
rootdir4='F:\提取_光谱分析\XX'
sta=cnn_train(cali_X,cali_Y,rootdir4)
T2=time.time()
print('程序运行时间:%s秒'%(T2-T1))
np.save('time.npy',T2-T1)
