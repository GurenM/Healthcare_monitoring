import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression

def datamaking(data):
    ECG = data.loc[:,'ECG'].values.reshape((-1,1))*4*1e-4
    AccZ = data.loc[:,'Acc Z'].values.reshape((-1,1))*5*1e-5
    u = np.hstack((ECG,AccZ))
    return u

def datamaking_posture(data):
    AccX = data.loc[:,'Acc X'].values.reshape((-1,1))*5*1e-5
    AccZ = data.loc[:,'Acc Z'].values.reshape((-1,1))*5*1e-5
    AccY = data.loc[:,'Acc Y'].values.reshape((-1,1))*5*1e-5

    u = np.hstack((AccX,AccY,AccZ))
    return u

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def softmax(x):
  if (x.ndim == 1):
    x = x[None,:]
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def trainmake(data,data2,data3,data4,data5):
    #Cut out the data in the areas that are not moving.
    #However, some of the data in the non-moving parts should also be included. 
    u_train = np.vstack((data[1000:2000],data[43000:46500],data[18000:19500],data[5500:8000],data[12000:15500],data[24500:27000],data[32000:37000],data[51000:71000]))
    u_train2 = np.vstack((data2[1000:2000],data2[43000:46500],data2[18000:19500],data2[5500:8000],data2[12000:15500],data2[24500:27000],data2[32000:37000]))
    u_train3 = np.vstack((data3[0:20750]))
    u_train4 = np.vstack((data4[1000:2000],data4[43000:46500],data4[18000:19500],data4[5500:8000],data4[12000:15500],data4[24500:27000],data4[32000:37000]))
    u_train5 = np.vstack((data5[1000:2000],data5[43000:46500],data5[18000:19500],data5[5500:8000],data5[12000:15500],data5[24500:27000],data5[32000:37000]))

    #Give gaussian noise on ECG sensor output
    np.random.seed(2)
    u_train6 = np.vstack((u_train[:,0]+np.random.normal(loc=0, scale=0.1, size=len(u_train[:,0])),u_train[:,1])).T
    u_train7 = np.vstack((u_train2[:,0]+np.random.normal(loc=0, scale=0.1, size=len(u_train2[:,0])),u_train2[:,1])).T
    u_train8 = np.vstack((u_train3[:,0]+np.random.normal(loc=0, scale=0.1, size=len(u_train3[:,0])),u_train3[:,1])).T
    u_train9 = np.vstack((u_train4[:,0]+np.random.normal(loc=0, scale=0.1, size=len(u_train4[:,0])),u_train4[:,1])).T
    u_train10 = np.vstack((u_train5[:,0]+np.random.normal(loc=0, scale=0.1, size=len(u_train5[:,0])),u_train5[:,1])).T

    #Create data with large amplitude values and add them to the training data.
    u_train11 = np.vstack((u_train[:1000,0]*6,u_train[:1000,1])).T
    u_train12 = np.vstack((u_train11,u_train11,u_train11,u_train11,u_train11))

    u_train = np.vstack((u_train,u_train2,u_train3,u_train4,u_train5,u_train6,u_train7,u_train8,u_train9,u_train10,u_train12))
    return u_train


def ESN_train(data,ResSize,input_magnitude,spectral_radius,leaking_rate):
    initLen = 1000 #washout length
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



def output(X_cough,u_posture,Wout_cough,Wout_posture):
    length_cough = len(X_cough[0,:])
    length_posture = len(u_posture[:,0])

    A = []
    Y_cough = np.zeros((length_cough,2))
    Y_cough_output = np.zeros(length_cough)

    Y_posture_proba = np.zeros((length_posture,5))
    Y_posture_output = np.zeros(length_posture)
    
    count = 0
    pass_len = 150
    for t in range(length_cough):
        Y_posture_proba[t,:] = (softmax((u_posture[t,:] @ Wout_posture[1:,:]) + Wout_posture[0,:]))
        for j in range(5):
            A.append(Y_posture_proba[t,j])        
        Y_posture_output[t] = np.argsort(A)[4]

        Y_cough[t,0] = np.round(sigmoid((X_cough[:,t].T @ Wout_cough[1:]) + Wout_cough[0]))
        if t < length_cough-30:
            if Y_cough[t,0] == 1 and Y_cough[t,1] ==0:
                for j in range(30):
                    Y_cough[t+j,0] = 1
                    Y_cough[t+j,1] = 1

        if t > pass_len:
            if Y_posture_output[t-1] != Y_posture_output[t]:
                count = pass_len
                for i in range(pass_len):
                    Y_cough_output[t-i] = 0 

            if count ==0:
                Y_cough_output[t] = Y_cough[t,1]
        
        if count > 0:
            count -=1

    return Y_cough_output, Y_posture_output

#load training data for cough detection
df_train_cough1= pd.read_excel('/.../.xlsx')
df_train_cough2 = pd.read_excel('/.../.xlsx')
df_train_cough3 = pd.read_excel('/.../.xlsx')
df_train_cough4 = pd.read_excel('/.../.xlsx')
df_train_cough5 = pd.read_excel('/.../.xlsx')
#load test data for cough detection
df_test = pd.read_excel('/.../.xlsx')

#Extract features
u_train_cough1 = datamaking(df_train_cough1) 
u_train_cough2 = datamaking(df_train_cough2) 
u_train_cough3 = datamaking(df_train_cough3) 
u_train_cough4 = datamaking(df_train_cough4) 
u_train_cough5 = datamaking(df_train_cough5)

#augmentation
u_train_cough = trainmake(u_train_cough1,u_train_cough2,u_train_cough3,u_train_cough4,u_train_cough5)
u_test_cough = datamaking(df_test)

#load target label for training data of cough detection
Yt_cough =  np.loadtxt('/.../.txt')
Yt_cough = Yt_cough[1000:u_train_cough.shape[0]] #1000 is washout length

#load target label for test data of cough detection
Y_cough_test = np.loadtxt('/.../.txt')

#load training data for posture detection
df_train_posture1 = pd.read_excel('/.../.xlsx')
df_train_posture2 = pd.read_excel('/.../.xlsx')

#extract features
u_train_posture1 = datamaking_posture(df_train_posture1)
u_train_posture2 = datamaking_posture(df_train_posture2)

#augmentation
u_train_posture = np.vstack((u_train_posture1,u_train_posture2))
u_test_posture = datamaking_posture(df_test)

#Definition:
#Stand up:0
#Face up:1
#Face left:2
#Face right:3
#Face down:4

#load target label for training data of posture detection
Yt_posture = np.loadtxt('/.../.txt')

#load target label for test data of posture detection
Y_posture_test = np.loadtxt('/.../.txt')



ResSize_ESN1 = 400
Input_magnitude_ESN1 = 1.25
spectral_radius_ESN1 = 0.1
leaking_rate_ESN1 = 0.1

ResSize_ESN2 = 20
Input_magnitude_ESN2 = 1.5
spectral_radius_ESN2 = 0.1
leaking_rate_ESN2 = 0.1

#ECG-drived Reservoir
X1_train = ESN_train(u_train_cough[:,0],ResSize_ESN1,Input_magnitude_ESN1,spectral_radius_ESN1,leaking_rate_ESN1)

#AccZ-drived Reservoir
X2_train = ESN_train(u_train_cough[:,1],ResSize_ESN2,Input_magnitude_ESN2,spectral_radius_ESN2,leaking_rate_ESN2)
X_train = np.vstack((X1_train,X2_train))

#calculate Wout
lr_cough = LogisticRegression(max_iter=10000)
lr_cough.fit(X_train.T,Yt_cough)
Wout_coef = lr_cough.coef_
Wout_intercept = lr_cough.intercept_

Wout_cough = np.zeros((5+ResSize_ESN1+ResSize_ESN2,1))
for i in range(5+ResSize_ESN1+ResSize_ESN2):
    if i == 0:
        Wout_cough[i,:] = Wout_intercept[:]
    if i >0:
        Wout_cough[i,:] = Wout_coef[:,i-1] 

#calculate Wout
lr_posture = LogisticRegression(max_iter=10000)
lr_posture.fit(u_train_posture,Yt_posture)
Wout_coef = lr_posture.coef_
Wout_intercept = lr_posture.intercept_
Wout_posture = np.zeros((4,5))

for i in range(4):
    if i == 0:
        Wout_posture[i,:] = Wout_intercept[:]
    if i >0:
        Wout_posture[i,:] = Wout_coef[:,i-1]  

#ECG-drived Reservoir
X1_test = ESN(u_test_cough[:,0],ResSize_ESN1,Input_magnitude_ESN1,spectral_radius_ESN1,leaking_rate_ESN1)
#AccZ-drived Reservoir
X2_test = ESN(u_test_cough[:,1],ResSize_ESN2,Input_magnitude_ESN2,spectral_radius_ESN2,leaking_rate_ESN2)
X_test_cough = np.vstack((X1_test,X2_test))

#output
Y_cough_output,Y_posture_output = output(X_test_cough,u_test_posture,Wout_cough,Wout_posture)

plt.figure(1).clear()
plt.plot(Y_cough_output,label="Predicted signal")
plt.plot(Y_cough_test+1.5, label = "Target signal")
plt.xlabel("Timestep")
plt.title("Cough detection")
plt.legend()

plt.figure(2).clear()
plt.plot(Y_posture_output,label="Predicted signal")
plt.plot(Y_posture_test+5, label = "Target signal")
plt.xlabel("Timestep")
plt.title("Posture detection")
plt.legend()
plt.show()
