import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression

def datamaking(data):
    ECG = data.loc[:,'ECG'].values.reshape((-1,1))*4*1e-4

    return ECG
    
def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def trainmake(data1,data2):
    np.random.seed(2)
    u_train1 = data1[:,0]
    u_train1 = np.hstack((u_train1[0:2600],u_train1[6400:9750]))
    u_train2 = data2[32000:37000,0]+np.random.normal(loc=0, scale=1, size=5000)

    u_train = np.hstack((u_train1,u_train2))
    return u_train

def ESN_train(data,ResSize,input_magnitude,spectral_radius,leaking_rate):
    initLen = 1000 #washout
    if data.ndim == 1:
        inSize = 1
    else:
        inSize = data.shape[1]
    datalength = data.shape[0]

    np.random.seed(42)
    Win_ESN1 = (np.random.rand(ResSize,1+inSize) - 0.5) * input_magnitude

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


def if_statement_detection(data,n=12):
    length = len(data)
    QRS = np.zeros((length,2))
    QRS_distance = np.zeros((length,2))

    pulse_distance_average = np.zeros((length))
    Arrhythmia = np.zeros((length))

    two_QRS_width_list = []
    two_pulse_distance_list = []
    summation_list = []

    current_pulse_distance = 0
    count_pulse_distance = 0
    count_QRS_width = 0
    
    average = 0
    count = 0
    for i in range(0,length-n+1):
        if i >=n:
            #standard deviation calculation
            #======================================================
            split_data = data[i-n:i]
            summation = 0
            for d in range(n):
                summation += np.square(np.average(split_data)-split_data[d])
            SD = np.sqrt((1/n)*summation)

            #QRS distance detection
            #======================================================
            count_pulse_distance +=1
            if SD >= 0.2:
                QRS[i,0] = 1
            else:
                QRS[i,0] = 0
            QRS[i,1] = SD

            if QRS[i,0] == 1 and QRS[i-1,0]==0:
                QRS_distance[i,0] = count_pulse_distance
                count_pulse_distance = 0
            if QRS[i,0] == 1:
                count_QRS_width +=1
            if QRS[i,0] == 0 and QRS[i-1,0]==1:
                QRS_distance[i,1] = count_QRS_width
                count_QRS_width = 0


            #classification
            #======================================================
            if QRS_distance[i,1] != 0:
                if len(two_QRS_width_list)<2:
                    two_QRS_width_list.append(QRS_distance[i,1])
                else:
                    del two_QRS_width_list[0]
                    two_QRS_width_list.append(QRS_distance[i,1])

            if len(two_pulse_distance_list)==2:
                current_pulse_distance +=1

            if QRS_distance[i,0] != 0:
                if len(two_pulse_distance_list)<2:
                    two_pulse_distance_list.append(QRS_distance[i,0])
                else:
                    del two_pulse_distance_list[0]
                    two_pulse_distance_list.append(QRS_distance[i,0])

                if len(two_pulse_distance_list)==2 and len(two_QRS_width_list)==2:
                    if count < 2 and two_pulse_distance_list[1]<1.1*two_pulse_distance_list[0] and two_pulse_distance_list[1]>0.9*two_pulse_distance_list[0] and two_pulse_distance_list[1]>30 and two_pulse_distance_list[0]>30:
                        count += 1
                        summation_list.append(QRS_distance[i,0])
                    elif count >= 2 and two_pulse_distance_list[1]<1.1*two_pulse_distance_list[0] and two_pulse_distance_list[1]>0.9*two_pulse_distance_list[0] and two_pulse_distance_list[1]>30 and two_pulse_distance_list[0]>30:
                        average = sum(summation_list)/2
                        summation_list.append(QRS_distance[i,0])
                        del summation_list[0]
                pulse_distance_average[i] = average
        
            if pulse_distance_average[i] != 0:
                if (two_pulse_distance_list[1] > pulse_distance_average[i]*1.25) and (two_pulse_distance_list[0] < pulse_distance_average[i]*0.75):
                    if two_QRS_width_list[0] < 30 and two_QRS_width_list[1] < 30:
                        for t in range(30):
                            Arrhythmia[i-current_pulse_distance+t-15] = 1
                current_pulse_distance = 0

    return Arrhythmia


def OR(ESN,condition):
    length = len(condition)
    result = np.zeros(length)

    for i in range(length):
        if ESN[i] == 1 and condition[i]==1:
            result[i] = 1
        elif ESN[i] == 0 and condition[i]==1:
            result[i] = 1
        elif ESN[i] == 1 and condition[i]==0:
            result[i] = 1
        elif ESN[i] == 0 and condition[i]==0:
            result[i] = 0

    return result

def output(X,Wout):
    length = len(X[0,:])
    Y1 = np.zeros((length,2))
    Y1[:,0] = np.round(sigmoid((X.T @ Wout[1:]) + Wout[0])).reshape(-1,)

    for i in range(length):
        if Y1[i,0] == 1 and Y1[i,1] ==0:
            for t in range(30):
                Y1[i-30+t,0] = 1
                Y1[i-30+t,1] = 1

    return Y1[:,1]
    
df_train1 = pd.read_excel('/.../.txt')
df_train2 = pd.read_excel('/.../.txt')
df_test = pd.read_excel('/.../.txt')

u_train1 = datamaking(df_train1)
u_train2 = datamaking(df_train2)

u_train = trainmake(u_train1,u_train2)
u_test = datamaking(df_test)

ResSize = 30
Input_magnitude = 1.7
spectral_radius = 0.1
leaking_rate = 0.1

X_train = ESN_train(u_train,ResSize,Input_magnitude,spectral_radius,leaking_rate)

Yt = np.loadtxt('/.../.txt')
Yt = Yt[1000:len(u_train)]

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train.T,Yt)

Wout_coef = lr.coef_
Wout_intercept = lr.intercept_
Wout = np.zeros((3+ResSize,1))
for i in range(3+ResSize):
    if i == 0:
        Wout[i,:] = Wout_intercept[:]
    if i >0:
        Wout[i,:] = Wout_coef[:,i-1] 

X_test = ESN(u_test,ResSize,Input_magnitude,spectral_radius,leaking_rate)
Y_test = np.loadtxt('/.../.txt')

Y_ESN = output(X_test,Wout)
Y_if = if_statement_detection(u_test)

if len(Y_ESN)<=len(Y_if):
    testlength = len(Y_ESN)
else:
    testlength = len(Y_ESN)

Y_output = OR(Y_ESN[:testlength],Y_if[:testlength])

plt.figure(1).clear()
plt.plot(Y_output,label="Predicted signal")
plt.plot(Y_test+1.5, label = "Target signal")
plt.xlabel("Timestep")
plt.title("Arrhythmia detection")
plt.legend()
plt.show()
