import tensorflow as tf 
from tensorflow import keras
from keras.layers import Dense, Dropout,LSTM
from keras.models import Sequential
import glob 
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
all_df = []
for file in (glob.glob("/Users/Vin/data/*.csv")):
    

    df=pd.read_csv(file,date_parser = True)
    
    all_df.append(df)
data = pd.concat(all_df,ignore_index = True)

rows = data[data['AwayTeam'] == 'Sivasspor']
row = data[data['HomeTeam'] == 'Sivasspor']

data = pd.concat([rows,row],ignore_index = True)

data= data.copy()

data["Date"] = pd.to_datetime(data["Date"],  dayfirst = True)

data = data.sort_values(by=['Date'])

del data["Time"]

data = data.dropna()


lb = preprocessing.LabelEncoder()

data["Div"] =lb.fit_transform(data["Div"])
data["Date"] = pd.to_datetime(data["Date"],  dayfirst = True)
data["HomeTeam"] =lb.fit_transform(data["HomeTeam"])
data["AwayTeam"] =lb.fit_transform(data["AwayTeam"])
data["FTR"] =lb.fit_transform(data["FTR"])

data = data.sort_values(by=['Date'])


data=data[['Div','Date','HomeTeam','AwayTeam','FTR','1','x','2']]
train = data[data['Date'] < "2022-5-1"].copy()
test = data[data['Date'] > "2022-5-1"].copy()


trainn = train.drop(['Date'],axis = 1)
scaler= MinMaxScaler()
train=scaler.fit_transform(trainn)

x_train=[]
y_train=[]

for i in range(5,train.shape[0]):
    x_train.append(train[i-5:i])
    y_train.append(train[i,3])  


x_train,y_train = np.array(x_train),np.array(y_train)

model = Sequential()

model.add(LSTM(128,input_shape =(x_train.shape[1],7),activation='relu',return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(18,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(56))
model.add(Dropout(0.2))

model.add(Dense(1,activation = 'softmax'))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3,decay=1e-5)

model.compile(optimizer = opt,loss = 'mean_squared_error',metrics=['binary_accuracy'])

pred=model.fit(x_train,y_train,epochs =50,batch_size =32,validation_data=(x_train,y_train))

last5 = trainn.tail(5)

df=last5.append(test,ignore_index=True)

df=df.drop(['Date'],axis=1)


input = scaler.fit_transform(df)

#print(input)

x_test=[]
y_test=[]

for i in range(5,input.shape[0]):
   x_test.append(input[i-5:i])
   y_test.append(input[i, 3])


x_test,y_test=np.array(x_test),np.array(y_test)

ypred=model.predict(x_test)

#print(scaler.scale_)

scale=1/0.5

y_pred = ypred*scale
y_test = y_test*scale

print(y_pred)
print(y_test)



