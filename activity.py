import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression

def datamaking(data):
    AccX = data.loc[:,'Acc X'].values.reshape((-1,1))*5*1e-5
    AccZ = data.loc[:,'Acc Z'].values.reshape((-1,1))*5*1e-5

    u = np.hstack((AccX,AccZ))
    return u

def trainmake(data,data1):
    #Cut out the data in the areas that are not moving.
    u_train = np.vstack((data[0:3000],data[6000:8000],data[24800:26500],data[20000:21000],data[32000:37000],data[12000:15500]))
    u_train2 = np.vstack((data1[:]))

    #Subtract 0.3 from AccZ
    u_train3 = np.vstack((u_train[:,0],u_train[:,1]-0.3)).T
    u_train4 = np.vstack((u_train2[:,0],u_train2[:,1]-0.3)).T

    #Add 0.3 on AccZ
    u_train5 = np.vstack((u_train[:,0],u_train[:,1]+0.3)).T
    u_train6 = np.vstack((u_train2[:,0],u_train2[:,1]+0.3)).T

    u_train = np.vstack((u_train,u_train2,u_train3,u_train4,u_train5,u_train6))
    return u_train

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def ESN_train(data,ResSize,input_magnitude,spectral_radius,leaking_rate):
    initLen = 1000 #washout
    if data.ndim == 1:
        inSize = 1
    else:
        inSize = data.shape[1]
    datalength = data.shape[0]

    np.random.seed(42)
    Win_ESN1 = (np.random.rand(ResSize,1+inSize) - 0.5) * input_magnitude
    np.random.seed(41)
    W1 = np.random.rand(ResSize,ResSize) - 0.5 
    rhoW = max(abs(linalg.eig(W1)[0]))
    W1 = W1 / rhoW*spectral_radius
    x1 = np.zeros((ResSize,1))
    X1 = np.zeros((1+inSize+ResSize,datalength-initLen))

    for t in range(datalength):
        if inSize == 1:
            u_current = data[t]
        else:
            u_current = data[t,:].reshape(inSize,1)
        x1 = (1-leaking_rate)*x1 + np.tanh( np.dot( Win_ESN1, np.vstack((1,u_current))) + np.dot( W1, x1 ))
        if t>initLen: 
            X1[:,t-initLen] = np.vstack((1,u_current,x1))[:,0]

    return X1

def ESN(data,ResSize,input_magnitude,spectral_radius,leaking_rate):
    if data.ndim == 1:
        inSize = 1
    else:
        inSize = data.shape[1]
    datalength = data.shape[0]

    np.random.seed(42)
    Win_ESN1 = (np.random.rand(ResSize,1+inSize) - 0.5) * input_magnitude
    np.random.seed(41)
    W1 = np.random.rand(ResSize,ResSize) - 0.5 
    rhoW = max(abs(linalg.eig(W1)[0]))
    W1 = W1 / rhoW*spectral_radius
    x1 = np.zeros((ResSize,1))
    X1 = np.zeros((1+inSize+ResSize,datalength))

    for t in range(datalength):
        if inSize == 1:
            u_current = data[t]
        else:
            u_current = data[t,:].reshape(inSize,1)
        x1 = (1-leaking_rate)*x1 + np.tanh( np.dot( Win_ESN1, np.vstack((1,u_current))) + np.dot( W1, x1 ))
        X1[:,t] = np.vstack((1,u_current,x1))[:,0]

    return X1

def output(X,Wout,smoothLength=125):
    count = 0
    length = len(X[0,:])
    Y1 = np.zeros((length,3))
    A = []
    for i in range(length):
        if i > 1000:
            Y1[i,0:2] = np.round(sigmoid((X[:,i].T @ Wout[1:,0]) + Wout[0,0]))
            
            if Y1[i,0] == 0 and Y1[i-1,0] ==1:
                Y1[i,2] = 1
            elif Y1[i,0] == 0:
                count +=1
            elif Y1[i,0] == 1 and Y1[i-1,0] ==0:
                
                #If there are two or more on states within smoothlength, interpolate on between them.
                for t in range(smoothLength):
                    if Y1[i-t,2] ==1:
                        A.append(True) 
                    else:
                        A.append(False)
                if any(A):
                    for t in range(count+2):
                        Y1[i-t,1] = 1
                count =0
                A = []
        
    return Y1[:,1]
    

#load training data
df_train1 = pd.read_excel('/.../.xlsx')
df_train2 = pd.read_excel('/.../.xlsx')
#load test data
df_test = pd.read_excel('/.../.xlsx')

#extract features
u_train1 = datamaking(df_train1)
u_train2 = datamaking(df_train2)
u_test = datamaking(df_test)
#augmentation
u_train = trainmake(u_train1,u_train2)

ResSize =40
Input_magnitude = 2.4 
spectral_radius =0.2
leaking_rate = 0.5

X_train = ESN_train(u_train,ResSize,Input_magnitude,spectral_radius,leaking_rate)

#load target label for training data
Yt = np.loadtxt('/.../.xlsx')
Yt = Yt[1000:len(u_train[:,0])] #1000 is washout

#calculate Wout
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train.T,Yt)

Wout_coef = lr.coef_
Wout_intercept = lr.intercept_
Wout = np.zeros((4+ResSize,1))

for i in range(4+ResSize):
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
plt.title("Activity detection")
plt.legend()