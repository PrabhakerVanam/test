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

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf

from numpy.random import seed

random_seed = 232323
seed(random_seed)
#from tensorflow import set_random_seed
#set_random_seed(random_seed)
tf.random.set_seed(random_seed)

data = pd.read_csv("D:\Prabhaker\Data Science\Keras\RA0034_predectors.csv")
#Let's check how many data points we have.
data.shape

#Let's check the dataset for any missing values.
data.describe()

data.isnull().sum()

#Split data into predictors and target

data.columns
''''UWI', 'TEST_DATE', 'OBJECTIVE', 'CHOKE', 'Choke (in)', 'WHP', 'FLP',
       'WHT', 'FLPT', 'OIL_RATE', 'WATER_RATE', 'GAS_RATE', 'GAS_RATE (MMSCF)',
       'GOR']
'''

data[['CHOKE','WHP','FLP','WHT']].plot(kind='hist')

predictors_df = data[['CHOKE','WHP','FLP','WHT']] 

target = data['OIL_RATE']

data[['CHOKE','WHP','FLP','WHT']].describe()

predictors_df.isnull().sum()
'''
CHOKE     0
WHP       0
FLP      11
WHT      11
'''


predictors_df.head()

#Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors_df - predictors_df.mean()) / (predictors_df.max()- predictors_df.min())
predictors_norm.head()

print(target.shape)

target.head()

#Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors_df.shape[1] # number of predictors

# Split the data in te Traiing and Testing set
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors_norm, target, test_size=0.3)

predictors_test.to_csv('RA0034.csv')

#predictors.to_csv('RA0034_predectors.csv')



# Building the Neural Network
def regression_model():
    #create model
    model = Sequential()
    model.add(Dense(10,activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10,activation='relu'))
    #model.add(Dense(16,activation='relu'))    
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and Test The model
# create model by calling the function
model = regression_model()

num_epochs = 200
batch_size = 8
validation_split = 0.20 # 30%

# Fit the model (train and test the model at the same time using the fit method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.)
history = model.fit(predictors_train, target_train, validation_split=validation_split, epochs=num_epochs, verbose=1)

#model.load_weights('d:\temp\best_weights.hdf5')

evalu_result = model.evaluate(predictors_test, target_test, batch_size=batch_size)
print(evalu_result)

pred_test = model.predict(predictors_test)

# new Data predict
#[['CHOKE','WHP','FLP','WHT']]
new_test_data =[[39,1719,339,189]]
new_pred_test = model.predict(pd.DataFrame(new_test_data, columns=['CHOKE','WHP','FLP','WHT']))

print("Shape: {}".format(pred_test.shape))
print(pred_test)

mean_square_error = mean_squared_error(target_test,pred_test)
# The mean squared error
print('Mean squared error: %.2f' % mean_square_error)

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(target_test, pred_test))

mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print("Mean Square error: ",mean)
print("Standard deviation", standard_deviation)


print_summary(model, line_length=None, positions=None, print_fn=None)