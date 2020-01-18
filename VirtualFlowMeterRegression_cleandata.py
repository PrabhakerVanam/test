# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:52:24 2020

@author: DELL
"""

import pandas as pd
import numpy as np

from keras.utils import print_summary

from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error, mean_squared_logarithmic_error
from keras import metrics
from keras import callbacks

#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn import metrics
#from sklearn.metrics import mean_squared_error, r2_score


import tensorflow as tf

from numpy.random import seed

random_seed = 232323
seed(random_seed)
#from tensorflow import set_random_seed
#set_random_seed(random_seed)
tf.random.set_seed(random_seed)

RA0034_data = pd.read_excel("D:\Prabhaker\Data Science\Keras\RA0034.xlsx")
#Let's check how many data points we have.
RA0034_data.shape

#Let's check the dataset for any missing values.
RA0034_data.describe()

RA0034_data.isnull().sum()

#Split data into predictors and target

RA0034_data.columns
''''UWI', 'TEST_DATE', 'OBJECTIVE', 'CHOKE', 'Choke (in)', 'WHP', 'FLP',
       'WHT', 'FLPT', 'OIL_RATE', 'WATER_RATE', 'GAS_RATE', 'GAS_RATE (MMSCF)',
       'GOR']
'''

RA0034_data[['CHOKE','WHP','FLP','WHT']].plot(kind='hist')

predictors_df = RA0034_data[['CHOKE','WHP','FLP','WHT']] 

target = RA0034_data['OIL_RATE']

RA0034_data[['CHOKE','WHP','FLP','WHT']].describe()

predictors_df.isnull().sum()
'''
CHOKE     0
WHP       0
FLP      11
WHT      11
'''
bins = [0,10,20,30,40,50,60,70,80]

predictors_df['BINNED'] = np.searchsorted(bins, predictors_df['CHOKE'].values)

# Get the average bin value
predictors_flp = predictors_df.groupby(['BINNED'])['FLP'].mean().reset_index()
predictors_flp = predictors_flp.set_index('BINNED')

predictors_wht = predictors_df.groupby(['BINNED'])['WHT'].mean().reset_index()
predictors_wht = predictors_wht.set_index('BINNED')

#predictors_flp.loc[3].tolist()[0]

# make a copy of dataframe
predictors = predictors_df[['CHOKE','WHP','FLP','WHT']]   

# Replace nan with average of same bin results (FLP)
predictors['FLP'] = predictors_df.apply(
    lambda row: predictors_flp.loc[row['BINNED']].tolist()[0] if np.isnan(row['FLP']) else row['FLP'],
    axis=1
)
# Replace nan with average of same bin results (WHT)
predictors['WHT'] = predictors_df.apply(
    lambda row: predictors_wht.loc[row['BINNED']].tolist()[0] if np.isnan(row['WHT']) else row['WHT'],
    axis=1
)

predictors_df.isnull().sum()
'''
CHOKE     0
WHP       0
FLP      11
WHT      11
'''
import seaborn as sns
sns.pairplot(predictors, diag_kind="kde")


#Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / (predictors.max()- predictors.min())
predictors_norm.head()

print(target.shape)

target.head()

#Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors.shape[1] # number of predictors

# Split the data in te Traiing and Testing set
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.2)

#predictors_test.to_csv('RA0034.csv')
predictors_status = predictors.describe()
predictors_status.pop("CHOKE")
predictors_status =predictors_status.transpose()
predictors_status

#predictors.to_csv('RA0034_predectors.csv')

# Building the Neural Network
def regression_model(dim, regress=False):
    #create model
    model = Sequential()
    model.add(Dense(16,activation='relu', input_dim=dim))
    model.add(Dense(8,activation='relu'))
     
    # check to see if the regression node should be added
    
    if regress :        
        model.add(Dense(1, activation="linear"))
    
    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    # compile model
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
   
    return model

# Train and Test The model
# create model by calling the function
model = regression_model(predictors.shape[1], regress=True)


    
num_epochs = 500
batch_size = 5
validation_split = 0.20 # 30%

# Fit the model (train and test the model at the same time using the fit method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.)
history = model.fit(predictors_train, target_train, validation_split=validation_split, epochs=num_epochs, verbose=2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

import matplotlib.pyplot as plt
plotter.plot({'Basic': history}, metric = "mae")

plotter.plot({'Basic': history}, metric = "mse")


model = regression_model(predictors.shape[1], regress=True)

# The patience parameter is the amount of epochs to check for improvement
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(predictors_train, target_train, 
                    epochs=num_epochs, validation_split = validation_split, verbose=1, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter.plot({'Early Stopping': early_history}, metric = "mae")

loss, mae, mse = model.evaluate(predictors_test, target_test, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))

pred_test = model.predict(predictors_test).flatten()
# convert to series
pred_test = pd.Series(pred_test.reshape(-1))
type(pred_test)

a = plt.axes(aspect='equal')
plt.scatter(target_test, pred_test)
plt.xlabel('True Values [OIL_RATE]')
plt.ylabel('Predictions [OIL_RATE]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = pred_test - target_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [OIL_RATE]")
_ = plt.ylabel("Count")

# new Data predict
#[['CHOKE','WHP','FLP','WHT']]
new_test_data =[[39,1719,339,189]]
new_pred_test = model.predict(pd.DataFrame(new_test_data, columns=['CHOKE','WHP','FLP','WHT']))

print("Shape: {}".format(pred_test.shape))
print(pred_test)

mean_square_error = mean_squared_error(target_test,pred_test)
# The mean squared error
print('Mean squared error: %.2f' % mean_square_error)

mean_square_logarithmic_error = mean_squared_logarithmic_error(target_test, pred_test)
# The mean squared error
print('Mean squared logarithmic error: %.2f' % mean_square_logarithmic_error)


acc = metrics.accuracy(target_test, pred_test)
print('Accuracy: {}'.format(acc))


#print_summary(model, line_length=None, positions=None, print_fn=None)