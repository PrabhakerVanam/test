# -*- coding: utf-8 -*-

##Import Keras and Packages
#mporting the keras libraries and the packages that we would need to build a neural network.
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


random_seed = 38388
from numpy.random import seed
seed(random_seed)
import tensorflow as tf
#from tensorflow import set_random_seed
#set_random_seed(random_seed)
tf.random.set_seed(random_seed)



vfm_data = pd.read_csv("D:\Prabhaker\Data Science\Keras\RA0034_predectors.csv", sep=',')

attributes = ['CHOKE','WHP','FLP','WHT', "OIL_RATE"]
#vfm_data.columns = attributes

vfm_data.head()

#Let's check how many data points we have.
vfm_data.shape


#Let's check the dataset for any missing values.
vfm_data.describe()

vfm_data.isnull().sum()


#Split data into predictors and targe
vfm_data_columns = vfm_data.columns
predictors = vfm_data.iloc[:,:4].astype(float) # all columns except class
target = vfm_data.iloc[:,4]# class column


#Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / (predictors.max()- predictors.min())
predictors_norm.head()

target.head()

print(target.shape)

#Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors.shape[1] # number of predictors


# Building the Neural Network
def regression_model():
    #create model
    model = Sequential()
    model.add(Dense(8,activation='relu', input_shape=(n_cols,)))
    #model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))    
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



# create model by calling the function
num_epochs = 150
batch_size = 5
estimator = KerasRegressor(build_fn=regression_model, epochs=num_epochs, batch_size=batch_size, verbose=1)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, predictors_norm, target, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

