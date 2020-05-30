# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:31:05 2020

@author: AD1006362
"""

#from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
import matplotlib.ticker as ticker
#from sklearn import preprocessing as pre
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
#import math
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score, cross_val_predict

from math import sqrt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6
#import statsmodels.api as sm
#from sklearn import metrics
#import random
#import csv

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 7777

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

def preprocess_data(file_name) :
    
    df_series = pd.read_csv(file_name, index_col ='TEST_DATE')
    #df_series =df_series.dropna()
    #df_series=df_series.reset_index(drop=True)
    df_series=df_series.iloc[:,2:13]
    print(df_series.describe())
    
    df_series = df_series.astype('float32')
    
    return df_series


def build_rf_model_oil_liq_gas(X, n_estimators=500, random_state=None):
    
    #K-fold validation
    scores = []
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    cv = KFold(n_splits=50, random_state=random_state, shuffle=True)
    for train_index, valid_index in cv.split(X):
        X_train, X_valid, Y_train, Y_valid = X[train_index], X[valid_index], Y[train_index], Y[valid_index]
        history = model.fit(X_train, Y_train)
        scores.append(model.score(X_valid, Y_valid))
        print(history)
    
    return model, scores

data = preprocess_data('RA0038GL_WT_Pro_LSTM.csv')

column_names=data.columns.values
print(column_names)

data1 = data.iloc[:,0:11]
# Preserv the date later to append to the dataframe
data1['Date'] = data1.index
data1.reset_index(drop=True, inplace=True)
data_id_date= data1.loc[:,'Date'] 

X= data.iloc[:,0:7]

Y= data.iloc[:,8:11]

X=np.array(X)
Y=np.array(Y)

scaler_X = MinMaxScaler(feature_range=(0, 1))
X = scaler_X.fit_transform(X)

scaler_Y = MinMaxScaler(feature_range=(0, 1))
Y = scaler_Y.fit_transform(Y)

random_state = seed_value

# Split data into Train and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=random_state)

# reshape input to be 3D [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Y_train= np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))
Y_test=  np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))


model, scores = build_rf_model_oil_liq_gas(X, n_estimators=700, random_state=random_state)
print(scores)

pred_test = model.predict(X_test.reshape(-1,7))

pred_train = model.predict(X_train.reshape(-1,7))

pyplot.plot(pred_test, label='Prediction')
pyplot.plot(Y_test, label='Test')
pyplot.legend()
pyplot.show()

#Mearn Absolute percentage
Errors = abs (pred_test - Y_test)
Mape = 100 * (Errors/Y_test)
Accuracy = 100 - np.mean(Mape)
print('Accuarcy:', round(Accuracy, 2), '%.')

#Calculate Coefficient of determination
r2 = r2_score(Y_test, pred_test)
print("R^2 for the fit is %f." % r2)


print("---------------------------------------------------------")
train_score = mean_squared_error(Y_train, pred_train)
print('Train Score: %.4f MSE' % (train_score))
test_score = mean_squared_error(Y_test,pred_test)
print('Test Score: %.4f MSE' % (test_score))
print("---------------------------------------------------------")

n_train = len(X_train)

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

fig.savefig("RF - Prediction quality on Train & Test.png")
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

fig.savefig("RF - Prediction quality on Test.png")
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
df_xy_train.to_csv('df_train_final_result_rf.csv', encoding='utf-8')
df_xy_test.to_csv('df_test_final_result_rf.csv', encoding='utf-8')


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
vPredict = model.predict(v_test_selected.reshape(-1,7))

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

v_final_df.to_csv('df_validate_final_result_rf.csv', encoding='utf-8')
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
fig.savefig("RF - Prediction quality on validation.png")
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




