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

#Let's check the dataset still any missing values.
predictors.isnull().sum()

predictors.describe()

# Plot histogram for null value columns 
plt.hist(predictors['FLP'], 4, normed=1, facecolor='green', alpha=0.75)
plt.hist(predictors['WHT'], 4, normed=1, facecolor='blue', alpha=0.75)

predictors.head()

#Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / (predictors.max()- predictors.min())
predictors_norm.head()

print(target.shape)

target.head()

#Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors.shape[1] # number of predictors

# Split the data in te Traiing and Testing set
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.3)

# Building the Neural Network
def regression_model():
    #create model
    model = Sequential()
    model.add(Dense(16,activation='relu', input_shape=(n_cols,)))
    model.add(Dense(16,activation='relu'))
    #model.add(Dense(8,activation='relu'))
    #model.add(Dense(16,activation='relu'))    
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and Test The model
# create model by calling the function
model = regression_model()

num_epochs = 800
batch_size = 5
validation_split = 0.20 # 30%

# Fit the model (train and test the model at the same time using the fit method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.)
history = model.fit(predictors_train, target_train, validation_split=validation_split, epochs=num_epochs, verbose=1)

#model.load_weights('d:\temp\best_weights.hdf5')

evalu_result = model.evaluate(predictors_test, target_test, batch_size=batch_size)
print(evalu_result)

pred_test = model.predict(predictors_test)
#my_list = map(lambda x: x[0], pred_test)
#pred_test = pd.Series(my_list)
pred_test = pd.Series(pred_test.reshape(-1))
type(pred_test)

print("Shape: {}".format(pred_test.shape))

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