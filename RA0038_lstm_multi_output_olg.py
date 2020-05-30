# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:31:05 2020

@author: AD1006362
"""

#from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.ticker as ticker
#from sklearn import preprocessing as pre
from sklearn.metrics import mean_squared_error, accuracy_score
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
#import math
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

from math import sqrt

#from keras.models import load_model

#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6
#import statsmodels.api as sm
#from sklearn import metrics
#import random
#import csv

def preprocess_data(file_name) :
    
    df_series = pd.read_csv(file_name)
   #df_series =df_series.dropna()
    df_series=df_series.reset_index(drop=True)
    df_series=df_series.iloc[:,3:14]
    print(df_series.describe())
    
    df_series = df_series.astype('float32')
    
    return df_series
#df_series = pd.read_csv('RA0038_Train_LSTM.csv')

def build_lstm_model_oil_liq_gas(input_shape=(50,6), return_sequences=True, num_outputs=3):
    
    regressor = Sequential()
    #regressor.add(LSTM(32, input_shape=(look_back, 1)))
    regressor.add(LSTM(units=64, activation='relu', return_sequences=return_sequences, input_shape=input_shape))
    regressor.add(Dropout(0.3))
    
    regressor.add(LSTM(units=128,activation='relu'))
    regressor.add(Dropout(0.5))
    
    regressor.add(Dense(units=num_outputs))
        
    regressor.summary()
    
    return regressor

data = preprocess_data('RA0038GL_WT_Pro_LSTM.csv')

column_names=data.columns.values
print(column_names)

data1 = data.iloc[:,0:11]
# Preserv the date later to append to the dataframe
data1['Date'] = data1.index
data1.reset_index(drop=True, inplace=True)
data_id_date= data1.loc[:,'Date'] 

# normalize features
X1= data.iloc[:,0:7]

Y= data.iloc[:,8:11]


X1=np.array(X1)
Y=np.array(Y)

scaler_X = MinMaxScaler(feature_range=(0, 1))
X1 = scaler_X.fit_transform(X1)

scaler_Y = MinMaxScaler(feature_range=(0, 1))
Y = scaler_Y.fit_transform(Y)

# Split data into Train and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=123)


# reshape input to be 3D [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Y_train= np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))
Y_test=  np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))


# Create LSTM model
model = build_lstm_model_oil_liq_gas(input_shape=(X_train.shape[1], X_train.shape[2]), num_outputs=3 )

# Compile model
start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse'])
print('compilation time : ', time.time() - start)

# Check point and save model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
es_patience=2
es = EarlyStopping(monitor="val_loss", mode="min", 
                       verbose=1, patience=es_patience)

# Add model checkpointing to save the best model
ckpt_file_path = os.path.join("model", "temp.hdf5")
ckpt = ModelCheckpoint(ckpt_file_path, monitor="val_loss", mode="min", 
                       verbose=1, save_best_only=True)

# Fit model
#regressor.fit(X_train, y_train, epochs=250, batch_size=64, verbose=1,  callbacks=[es, ckpt])  
#history = regressor.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[es, ckpt])
history=model.fit(X_train, Y_train, epochs=300, batch_size=32, validation_data=(X_test, Y_test),verbose=1)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Predict model
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


print("---------------------------------------------------------")
train_score = mean_squared_error(Y_train, pred_train)
print('Train Score: %.4f MSE' % (train_score))
test_score = mean_squared_error(Y_test,pred_test)
print('Test Score: %.4f MSE' % (test_score))
print("---------------------------------------------------------")

n_train=X_train.shape[0]


# Plot on Train Data
fig, (ax1, ax2, ax3) = pyplot.subplots(nrows=3, ncols=1, sharex=True)
fig.tight_layout(pad=3.0)
print("---------------------------------------------------------")
ax1.set_title('Prediction quality Oil')
ax1.plot(range(n_train), Y_train[:,0], label="Train")
ax1.plot(range(n_train, len(Y_test) + n_train), Y_test[:,0], '-', label="Test")
ax1.plot(range(n_train, len(Y_test) + n_train), pred_test[:,0], '--',label="Prediction test")
ax1.legend(loc=(1.01, 0))

ax2.set_title('Prediction quality Liquid')
ax2.plot(range(n_train), Y_train[:,1], label="Train")
ax2.plot(range(n_train, len(Y_test) + n_train), Y_test[:,1], '-', label="Test")
ax2.plot(range(n_train, len(Y_test) + n_train), pred_test[:,1], '--',label="Prediction test")
ax2.legend(loc=(1.01, 0))

ax3.set_title('Prediction quality GAS')
ax3.plot(range(n_train), Y_train[:,2], label="Train")
ax3.plot(range(n_train, len(Y_test) + n_train), Y_test[:,2], '-', label="Test")
ax3.plot(range(n_train, len(Y_test) + n_train), pred_test[:,2], '--',label="Prediction test")
ax3.legend(loc=(1.01, 0))

fig.savefig("LSTM - Prediction quality on Train & Test.png")
print("---------------------------------------------------------")

# Plot on Test Data
fig, (ax1, ax2, ax3) = pyplot.subplots(nrows=3, ncols=1, sharex=True)
fig.tight_layout(pad=3.0)
print("---------------------------------------------------------")
ax1.set_title('Prediction quality Oil')
ax1.plot(Y_test.reshape(-1, 3)[:,0],'-', label="Observed")
ax1.plot(pred_test.reshape(-1, 3)[:,0], '--',label="Prediction")
ax1.legend(loc=(1.01, 0))

ax2.set_title('Prediction quality Liquid')
ax2.plot(Y_test.reshape(-1, 3)[:,1],'-', label="Observed")
ax2.plot(pred_test.reshape(-1, 3)[:,1], '--',label="Prediction")
ax2.legend(loc=(1.01, 0))

ax3.set_title('Prediction quality GAS')
ax3.plot(Y_test.reshape(-1, 3)[:,2],'-', label="Observed")
ax3.plot(pred_test.reshape(-1, 3)[:,2], '--',label="Prediction")
ax3.legend(loc=(1.01, 0))

fig.savefig("LSTM - Prediction quality on Test.png")
print("---------------------------------------------------------")
# Reshape the data back to original (Training Data set)
xtrain = X_train.reshape(-1,X_train.shape[1])
ytrain = Y_train.reshape(-1,3)
ptrain = pred_train.reshape(-1,3)


# invers transform
xtrain = scaler_X.inverse_transform(xtrain)
ytrain = scaler_Y.inverse_transform(ytrain)
ptrain = scaler_Y.inverse_transform(ptrain)

dfx = pd.DataFrame(xtrain)
dfy = pd.DataFrame(ytrain)
dfpy = pd.DataFrame(ptrain)

# merge data frame for Train set
df_xy_train = pd.concat([dfx,dfy,dfpy], axis=1)

# Final data frame columns
col_names_all = ['CHOKE_SIZE', 'TUBING_PRESSURE', 'TUBING_TEMPERATURE',
       'FLOWLINE_PRESSURE', 'FLOWLINE_TEMPERATURE', 'Choke_DP',
       'GAS_LIFT_VOLUME', 'OIL_RATE', 'LIQUID_RATE', 'GAS_RATE', 'PRD_OIL_RATE', 'PRD_LIQUID_RATE', 'PRD_GAS_RATE']
# set column names on Train Dataframe
df_xy_train.columns =col_names_all

# Reshape the data back to original (Testing Data set)
xtest = X_test.reshape(-1,X_test.shape[1])
ytest = Y_test.reshape(-1,3)
ptest = pred_test.reshape(-1,3)

# invers transform
xtest = scaler_X.inverse_transform(xtest)
ytest = scaler_Y.inverse_transform(ytest)
ptest = scaler_Y.inverse_transform(ptest)

dftx = pd.DataFrame(xtest)
dfty = pd.DataFrame(ytest)
dftpy = pd.DataFrame(ptest)

# merge data frame for Test set
df_xy_test = pd.concat([dftx,dfty,dftpy], axis=1)

# set column names on Train Dataframe
df_xy_test.columns =col_names_all

# Save Final Train and Test dataset back to CSV
df_xy_train.to_csv('df_train_final_result_lstm.csv', encoding='utf-8')
df_xy_test.to_csv('df_test_final_result_lstm.csv', encoding='utf-8')


# Predict values based on new dataset
# Data Preparation
v_test = pd.read_csv("RA0038_AvgRT_LSTM.csv", index_col ='Date', parse_dates=True)
#v_test['Production_Date']=pd.to_datetime(v_test['Production_Date'])
#v_test = v_test[v_test['Production_Date']<"9/17/2019 0:00"]

# Preserv the date later to append to the dataframe
v_test_id_date = v_test.iloc[:,1:11]
v_test_id_date['Date'] = v_test_id_date.index
v_test_id_date.reset_index(drop=True, inplace=True)

v_test_selected= v_test_id_date.iloc[:,0:7]
v_test_selected = scaler_X.fit_transform(v_test_selected)
v_test_selected=np.array(v_test_selected)
v_test_selected = np.reshape(v_test_selected, (v_test_selected.shape[0], v_test_selected.shape[1], 1))

# Predict 
vPredict = model.predict(v_test_selected.reshape(-1,7,1))

# reshape Predicted values back to 3 outputs
vPredict = vPredict.reshape(-1,3)
vPredict = scaler_Y.inverse_transform(vPredict)
dfvpy = pd.DataFrame(vPredict)

# merge data frame for Validate set
v_final_df = pd.concat([v_test_id_date.iloc[:,0:11],dfvpy], axis=1)

# Set Date is the index column
v_final_df.set_index('Date', drop=True, inplace=True)
v_final_df.index = v_final_df.index.strftime('%Y/%m/%d')
# set column names on Train Dataframe
v_final_df.columns = col_names_all

# Save Final validated/predicted dataset back to CSV

v_final_df.to_csv('df_validate_final_result_lstm.csv', encoding='utf-8')
print("---------------------------------------------------------")


# sort dataframe by index descend (Date)
v_final_df_sorted = v_final_df.sort_index(axis=0, ascending=True, inplace=False)


# Plot on Test Data
label_size = 12
pyplot.rcParams['xtick.labelsize'] = label_size 
fig, (ax1, ax2, ax3) = pyplot.subplots(nrows=3, ncols=1, sharex=True)
fig.tight_layout(pad=3.0)

headcount = len(v_final_df)
#if len(v_final_df) < 100 :
#    headcount = len(v_final_df)
    
print("---------------------------------------------------------")
ax1.set_title('Prediction quality Oil')
ax1.plot(v_final_df_sorted.iloc[:,7].head(headcount), label="Observed")
ax1.plot(v_final_df_sorted.iloc[:,10].head(headcount), label="Prediction")
ax1.legend(loc=(1.01, 0))
ax1.xaxis.set_major_locator(ticker.LinearLocator(10))
ax1.xaxis.set_ticks_position('bottom')

ax2.set_title('Prediction quality Liquid')
ax2.plot(v_final_df_sorted.iloc[:,8].head(headcount), label="Observed")
ax2.plot(v_final_df_sorted.iloc[:,11].head(headcount),label="Prediction")
ax2.legend(loc=(1.01, 0))
ax2.xaxis.set_major_locator(ticker.LinearLocator(10))
ax2.xaxis.set_ticks_position('bottom')

ax3.set_title('Prediction quality GAS')
ax3.plot(v_final_df_sorted.iloc[:,9].head(headcount), label="Observed")
ax3.plot(v_final_df_sorted.iloc[:,12].head(headcount),label="Prediction")
ax3.legend(loc=(1.01, 0))
ax3.xaxis.set_major_locator(ticker.LinearLocator(10))
ax3.xaxis.set_ticks_position('bottom')

fig.autofmt_xdate(rotation=45)
fig.savefig("LSTM - Prediction quality on validation.png")
print("---------------------------------------------------------")

Errors = abs (v_final_df_sorted.iloc[:,7] - v_final_df_sorted.iloc[:,10])
Mape = 100 * (Errors/v_final_df_sorted.iloc[:,7])
Accuracy = 100 - np.mean(Mape)
print('Accuracy OIL :', round(Accuracy, 2), '%.')

Errors = abs (v_final_df_sorted.iloc[:,8] - v_final_df_sorted.iloc[:,11])
Mape = 100 * (Errors/v_final_df_sorted.iloc[:,8])
Accuracy = 100 - np.mean(Mape)
print('Accuracy LIQUID :', round(Accuracy, 2), '%.')

Errors = abs (v_final_df_sorted.iloc[:,9] - v_final_df_sorted.iloc[:,12])
Mape = 100 * (Errors/v_final_df_sorted.iloc[:,8])
Accuracy = 100 - np.mean(Mape)
print('Accuracy GAS :', round(Accuracy, 2), '%.')

rmse_validate = sqrt(mean_squared_error(v_final_df.iloc[:,7:10], v_final_df.iloc[:,10:]))
print('Validate RMSE: %.3f' % rmse_validate)
print("---------------------------------------------------------")


# Run Below code only for TEST OR Validate new data without training ( load model from disk)

model_file_path = os.path.join("model", "ra38_oil_liq_gas_model.hdf5")
model.save(model_file_path)

#model = load_model(model_file_path)



