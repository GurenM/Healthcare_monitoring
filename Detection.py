import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

def softmax(x):
  if (x.ndim == 1):
    x = x[None,:]
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def datamaking(data):
    AccX = data.loc[:,'Acc X'].values.reshape((-1,1))*5*1e-5
    AccZ = data.loc[:,'Acc Z'].values.reshape((-1,1))*5*1e-5
    AccY = data.loc[:,'Acc Y'].values.reshape((-1,1))*5*1e-5

    u = np.hstack((AccX,AccY,AccZ))
    return u


def output(X,Wout):
    length = len(X[:,0])
    Y1 = np.zeros((length,5))
    output = np.zeros((length,1))
    A = []
    for i in range(length):
        Y1[i,:] = (softmax((X[i,:] @ Wout[1:,:]) + Wout[0,:]))
        for j in range(5):
            A.append(Y1[i,j])
        output[i,0] = np.argsort(A)[4]
        A = []
            
    return output 

def onehot_4dim(data):
    dataLength = data.shape[0]
    fourdim_data = np.zeros((dataLength,5))
    for i in range(dataLength):
        for j in range(5):
            if data[i] == j:
                fourdim_data[i,j] = 1
            
    return fourdim_data

def accuracy(Y1,Y_test):        
    count_st: int = 0
    count_up: int = 0
    count_right: int = 0
    count_left: int = 0
    count_down: int = 0
    testLen= len(Y_test[:])
    Accuracy = np.arange(25).reshape((5, 5))
    for j in range(5):
        for t in range(testLen):
            if Y_test[t] == j and Y1[t] == 0:
                count_st +=1
            elif Y_test[t] == j and Y1[t] ==1:
                count_up +=1
            elif Y_test[t] == j and Y1[t] ==2:
                count_left +=1
            elif Y_test[t] == j and Y1[t] ==3:
                count_right +=1
            elif Y_test[t] == j and Y1[t] ==4:
                count_down +=1
        Accuracy[0,j] = int(count_st)
        Accuracy[1,j] = int(count_up)
        Accuracy[2,j] = int(count_left)
        Accuracy[3,j] = int(count_right)
        Accuracy[4,j] = int(count_down)
        count_st = 0
        count_up = 0
        count_right = 0
        count_left = 0
        count_down = 0

    return Accuracy

df_train1 = pd.read_excel('excel/Lyingdown_revised.xlsx',engine='openpyxl').loc[1000:38890]
df_train2 = pd.read_excel('excel/volunteer13.xlsx',engine='openpyxl').loc[43000:71200]
df_test = pd.read_excel('excel/volunteer5_0621_revised.xlsx',engine='openpyxl')

u_train = datamaking(df_train1)
u_train2 = datamaking(df_train2)
u_train = np.vstack((u_train,u_train2))
u_test = datamaking(df_test)

trainLen = u_train.shape[0]
testLen = u_test.shape[0]

#Definition:
#Stand up:0
#Face up:1
#Face left:2
#Face right:3
#Face down:4
Yt =np.hstack((np.zeros(2770),np.ones(6670-2770),np.full(10630-6670,2),np.ones(14480-10630),np.full(18400-14480,3),
                np.ones(22220-18400),np.full(26250-22220,4),np.ones(30375-26250),np.zeros(37892-30375),
                np.zeros(44732-37891),np.ones(52207-44732),np.full(58984-52207,3),np.full(66091-58984,2)))

Y_test = np.hstack((np.zeros(50456),np.ones(54955-50456),np.full(59846-54955,3),np.ones(60104-59846),np.full(65051-60104,2),
                np.full(70000-65051,4)))

lr = LogisticRegression(max_iter=3000)
lr.fit(u_train[:trainLen],Yt)
Wout_coef = lr.coef_
Wout_intercept = lr.intercept_
Wout = np.zeros((4,5))

for i in range(4):
    if i == 0:
        Wout[i,:] = Wout_intercept[:]
    if i >0:
        Wout[i,:] = Wout_coef[:,i-1]  

Y_output = output(u_test,Wout)

plt.plot(Y_output)
plt.plot(Y_test+5)
plt.show()