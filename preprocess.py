import numpy as np
import pandas as pd

def trunc(values, decs):
    return np.trunc(values*10**decs)/(10**decs)

#load raw data
df = pd.read_excel('\..\....')

#specify measurement end time and number of data
LastTime = 589.9919 #arbitary number
datasize = 73116 #arbitary number

time = np.arange(0,LastTime+0.008,LastTime/datasize) 
frame1 = pd.DataFrame(df,columns=["Acc X","Acc Y","Acc Z","R1","R2"],index=time)
frame2 = pd.DataFrame(df,columns=["Acc X","Acc Y","Acc Z","R1","R2"])

X = []

for i in np.arange(0.0000,LastTime,0.100):
    df1 = frame1[i:i+0.1].fillna(frame2.loc[int(i*10)]) 
    X.append(df1)

df_con = pd.concat(X)
df_con.to_excel("\...\....")
