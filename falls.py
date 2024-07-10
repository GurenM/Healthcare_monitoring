import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression


def datamaking(data):
    AccZ = data.loc[:,'Acc Z'].values.reshape((-1,1))*5*1e-5

    return AccZ


def trainmake(data,data1):
    #Cut out the data in the areas that are not moving.
    u_train = np.vstack((data[0:2250],data[6600:11292]))
    u_train2 = np.vstack((data1[0:3000],data1[6000:8000],data1[24800:26500],data1[20000:21000],data1[32000:37000],data1[12000:15500]))
    u_train = np.vstack((u_train,u_train2))
    return u_train

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def ESN_train(data1,Ressize,input_magnitude,spec,a):
    initLen = 1000 #washout
    if data1.ndim == 1:
        inSize = 1
    else:
        inSize = data1.shape[1]
    datalen = data1.shape[0]
    np.random.seed(42)
    Win_ESN1 = (np.random.rand(Ressize,1+inSize) - 0.5) * input_magnitude
    np.random.seed(41)
    W1 = np.random.rand(Ressize,Ressize) - 0.5 
    rhoW = max(abs(linalg.eig(W1)[0]))
    W1 = W1 / rhoW*spec
    x1 = np.zeros((Ressize,1))
    X1 = np.zeros((1+inSize+Ressize,datalen-initLen))

    for t in range(datalen):
        if inSize == 1:
            u_current = data1[t]
        else:
            u_current = data1[t,:].reshape(inSize,1)
        x1 = (1-a)*x1 + np.tanh( np.dot( Win_ESN1, np.vstack((1,u_current))) + np.dot( W1, x1 ))
        if t>initLen: 
            X1[:,t-initLen] = np.vstack((1,u_current,x1))[:,0]

    return X1

def ESN(data1,Ressize,input_magnitude,spec,a):
    if data1.ndim == 1:
        inSize = 1
    else:
        inSize = data1.shape[1]
    datalen = data1.shape[0]
    np.random.seed(42)
    Win_ESN1 = (np.random.rand(Ressize,1+inSize) - 0.5) * input_magnitude
    np.random.seed(41)
    W1 = np.random.rand(Ressize,Ressize) - 0.5 
    rhoW = max(abs(linalg.eig(W1)[0]))
    W1 = W1 / rhoW*spec
    x1 = np.zeros((Ressize,1))
    X1 = np.zeros((1+inSize+Ressize,datalen))

    for t in range(datalen):
        if inSize == 1:
            u_current = data1[t]
        else:
            u_current = data1[t,:].reshape(inSize,1)
        x1 = (1-a)*x1 + np.tanh( np.dot( Win_ESN1, np.vstack((1,u_current))) + np.dot( W1, x1 ))
        X1[:,t] = np.vstack((1,u_current,x1))[:,0]

    return X1

def output(X,Wout):
    length = len(X[0,:])
    Y1 = np.zeros((length,2))
    Y1[:,0] = np.round(sigmoid((X.T @ Wout[1:]) + Wout[0])).reshape(-1,)
    
    for i in range(length-30):
        if i > 1000:
            if Y1[i,0] == 1 and Y1[i,1] ==0:
                for t in range(30):
                    Y1[i+t,0] = 1
                    Y1[i+t,1] = 1
    
    
    return Y1[:,1]
     
#load trainig data
df_train1=pd.read_excel('/.../.xlsx')
df_train2 = pd.read_excel('/.../.xlsx')

#load test data
df_test = pd.read_excel('/.../.xlsx')

#extract features
u_train1 = datamaking(df_train1)
u_train2 = datamaking(df_train2)
u_test = datamaking(df_test)

#augmentation
u_train = trainmake(u_train1,u_train2)


ResSize = 50 
Input_magnitude = 0.8
spectral_radius = 0.3
leaking_rate = 0.3

X_train = ESN_train(u_train,ResSize,Input_magnitude,spectral_radius,leaking_rate)

#loat target label for training data
Yt =np.loadtxt('/.../.xlsx')
Yt = Yt[1000:len(u_train[:,0])]

#calculate Wout
lr1 = LogisticRegression(max_iter=10000)
lr1.fit(X_train.T,Yt)
Wout_coef = lr1.coef_
Wout_intercept = lr1.intercept_

Wout = np.zeros((3+ResSize,1))

for i in range(3+ResSize):
    if i == 0:
        Wout[i,:] = Wout_intercept[:]
    if i >0:
        Wout[i,:] = Wout_coef[:,i-1] 

X_test = ESN(u_test,ResSize,Input_magnitude,spectral_radius,leaking_rate)

#load target label for test data
Y_test =  np.loadtxt('/.../.xlsx')

#output
Y_output = output(X_test,Wout)

plt.figure(1).clear()
plt.plot(Y_output,label="Predicted signal")
plt.plot(Y_test+1.5, label = "Target signal")
plt.xlabel("Timestep")
plt.title("Falls detection")
plt.legend()