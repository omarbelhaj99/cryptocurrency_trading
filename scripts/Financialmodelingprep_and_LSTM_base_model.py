#!/usr/bin/env python
# coding: utf-8

# # scrapping daily bts prices

# In[1]:


import requests
import pandas as pd


# In[2]:


api_key='3c63db1acd4399020c7f6eee92ec77cf'
symbol='BTCUSD'
query=f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'


# In[3]:


response=requests.get(query)


# In[4]:


btsdaily=pd.DataFrame(response.json()['historical'])


# In[5]:


btsdaily


# In[6]:


btsdaily.to_csv('btsdaily', encoding='utf-8', index=False)


# In[7]:


pd.read_csv('btsdaily')


# # basline model using only historical data

# In[8]:


import math
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# In[29]:


df=pd.read_csv('btsdaily')[::-1].reset_index()
df=df.drop('index',axis=1)
df


# In[30]:


training_set = df.iloc[:1276, 1:2].values
test_set = df.iloc[1276:, 1:2].values


# ## scaling

# In[59]:


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, 1276):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# ## model sequential 

# In[32]:


model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)


# ## Prepare test Data

# In[56]:


dataset_train = df.iloc[:1276, 1:2]
dataset_test = df.iloc[1276:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,551):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)


# In[57]:


predicted_bts_price = model.predict(X_test)
predicted_bts_price = sc.inverse_transform(predicted_bts_price)
predicted_bts_price.shape


# In[78]:


predicted_bts_price.shape


# ## Plot

# In[80]:


# Visualising the results
plt.plot(df.loc[1276:, 'date'], dataset_test.values, color = 'red', label = 'Real BTS Price')
plt.plot(df.loc[1336:, 'date'], predicted_bts_price, color = 'blue', label = 'Predicted BTS Price')
plt.xticks(np.arange(0,491,50))


plt.title('BTS Prediction')
plt.xlabel('Time')
plt.ylabel('BTS Price')
plt.legend()
plt.show()


# ## MSE

# In[86]:


MSE= sum((dataset_test.values[60:]-predicted_bts_price)**2)/len(predicted_bts_price)
MSE


# In[ ]:




