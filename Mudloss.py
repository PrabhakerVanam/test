# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:47:47 2020

@author: DELL
"""

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error, mean_squared_logarithmic_error
from keras import metrics
from keras import callbacks

random_seed = 45454545
from numpy.random import seed
seed(random_seed)

# =============================================================================
# import chardet
#  
# data_base_path="D:\\Prabhaker\\Data Science\\Keras\MUDLOSS\\"
# 
# with open(data_base_path+'BAB Data1.csv', 'rb') as f:
# 
#     result = chardet.detect(f.read())
#  
# 
# babdata1 = pd.read_csv(data_base_path+'BAB Data1.csv', encoding=result['encoding'])
# 
# babdata2 = pd.read_csv(data_base_path+'BAB Data2.csv', encoding=result['encoding'])
# 
# babdata3 = pd.read_csv(data_base_path+'BAB Data3.csv', encoding=result['encoding'])
# 
# babdata41 = pd.read_csv(data_base_path+'BAB Data4.1.csv', encoding=result['encoding'])
# 
# babdata42 = pd.read_csv(data_base_path+'BAB Data4.2.csv', encoding=result['encoding'])
# 
# babdata43 = pd.read_csv(data_base_path+'BAB Data4.3.csv', encoding=result['encoding'])
# 
# babdata44 = pd.read_csv(data_base_path+'BAB Data4.4.csv', encoding=result['encoding'])
# 
# 
# babdata4 = babdata41.append(babdata42).append(babdata43).append(babdata44)
# 
# 
# finaldata = pd.merge(babdata1, babdata2, how='outer' )
# 
# finaldata = pd.merge(finaldata, babdata3, how='outer' )
# 
# finaldata = pd.merge(finaldata, babdata4, how='outer' )
#  
# finaldata.to_csv(data_base_path+'BAB Data - Merged.csv')
# 
# finaldata.head()
# =============================================================================

df = pd.read_csv('D:\\Prabhaker\\Data Science\\Keras\MUDLOSS\\BAB Data - Merged.csv')

df.head()

df.describe()

df.shape

df.isnull().sum()

print(df.columns)

muddata=df.copy()
muddata.drop(columns=['Unnamed: 0'], inplace=True)

#muddata_loss = muddata[np.isfinite(muddata['Mud Loss (bbl)'])]
muddata_loss = muddata[pd.notnull(muddata['Mud Loss (bbl)'])]

print(muddata_loss.columns)

columns = muddata_loss.columns
i = 0
newcolumns=[]
while i < len(columns):
    newcolumns.append(columns[i].strip())
    i += 1

print(newcolumns)
# assign stripped column names
muddata_loss.columns=newcolumns


muddata_loss.set_index(['Report date','Well'], inplace=True, drop = False)

#muddata_loss = muddata_loss.reset_index(level=1, drop=True, inplace=False, col_level=1)

#muddata_loss.drop(columns=['Unnamed: 0'], inplace=True)

print(muddata_loss.shape)

print(muddata_loss.info())

columns_floattype =['Mid Night Depth (ft)','Avg. ROP (ft/hr)','Flow rate (gpm)','Min. RPM (rpm)','Max. RPM (rpm)','Max. WOB (kip)','Min. WOB (kip)','TFA (in²)', 'ECD (lbm/ft³)','Density (ppg)','PV (cp)','YP (lbf/100ft²)','Mud Loss (bbl)','Subsurface total (bbl)','SPM (spm)','Pressure (psi)']
# Clean the data
for column in columns_floattype:
    print(column)
    if muddata_loss[column].dtype != 'float64':
        muddata_loss= muddata_loss.replace({column: {",": ""}}, regex=True)
       
#muddata_loss[[columns_floattype]] = muddata_loss[[columns_floattype]].replace({'Min. RPM (rpm)': {",": ""}}, regex=True)



# =============================================================================
# muddata_loss= muddata_loss.replace({'Min. RPM (rpm)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Max. RPM (rpm)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Min. WOB (kip)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Max. WOB (kip)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'TFA (in²)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'ECD (lbm/ft³)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Mid Night Depth (ft)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Avg. ROP (ft/hr)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Flow rate (gpm)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Density (ppg)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'PV (cp)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'YP (lbf/100ft²)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Mud Loss (bbl)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Density (ppg)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Subsurface total (bbl)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'SPM (spm)': {",": ""}}, regex=True)
# muddata_loss= muddata_loss.replace({'Pressure (psi)': {",": ""}}, regex=True)
# =============================================================================




# =============================================================================
# muddata_comments = muddata_loss['Comments']
# 
# muddata_loss['Comments'].dtype
# dictionary = {',':''}
# #muddata_loss=muddata_loss.replace(dictionary, regex=False)
# muddata_loss= muddata_loss.applymap(lambda x: x if type(x)!='str' else x.replace(","," ") )
# muddata_loss['Comments']=muddata_comments
# =============================================================================

# Convert datatypes to float
for column in columns_floattype:
    print(column)
    if muddata_loss[column].dtype != 'float64':
        muddata_loss[column] =  muddata_loss[column].astype('float64')
# =============================================================================
# muddata_loss['Min. RPM (rpm)'] =muddata_loss['Min. RPM (rpm)'].astype('float64')
# muddata_loss['Max. RPM (rpm)'] =muddata_loss['Max. RPM (rpm)'].astype('float64')
# muddata_loss['Min. WOB (kip)'] =muddata_loss['Max. WOB (kip)'].astype('float64')
# muddata_loss['Max. WOB (kip)'] =muddata_loss['Max. WOB (kip)'].astype('float64')
# =============================================================================

print(muddata_loss.isnull().sum())

print(muddata_loss.describe())

print(muddata_loss.dtypes)

print(muddata_loss.head())

# Fill nan with average values
#muddata_loss = muddata_loss.apply(lambda x: x.fillna(x.mean(), axis=0))
for column in columns_floattype:
    muddata_loss[column] = muddata_loss[column].fillna(muddata_loss[column].mean(), axis=0)

# make mud loss colum as last column in the data frame
mloss=muddata_loss['Mud Loss (bbl)']
muddata_loss.drop('Mud Loss (bbl)', axis=1, inplace=True)
muddata_loss['Mud Loss (bbl)'] = mloss

# select featured columns ['Formation']
predictors=muddata_loss[['Mid Night Depth (ft)','Avg. ROP (ft/hr)','Flow rate (gpm)', 'Min. RPM (rpm)', 'Max. RPM (rpm)', 'Max. WOB (kip)', 'Min. WOB (kip)', 'TFA (in²)', 'ECD (lbm/ft³)','Density (ppg)', 'PV (cp)', 'YP (lbf/100ft²)', 'Subsurface total (bbl)','SPM (spm)','Pressure (psi)']]
target=muddata_loss['Mud Loss (bbl)']


#Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / (predictors.max()- predictors.min())
predictors_norm.head()

target.head()

print(target.shape)

# Split the data in te Traiing and Testing set
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors_norm, target, test_size=0.2)


#Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors.shape[1] # number of predictors


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


    
num_epochs = 200
batch_size = 20
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
plt.xlabel('True Values [MUD LOSS]')
plt.ylabel('Predictions [MUD LOSS]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = pred_test - target_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MUD LOSS]")
_ = plt.ylabel("Count")


mean_square_error = mean_squared_error(target_test,pred_test)
# The mean squared error
print('Mean squared error: %.2f' % mean_square_error)

mean_square_logarithmic_error = mean_squared_logarithmic_error(target_test, pred_test)
# The mean squared error
print('Mean squared logarithmic error: %.2f' % mean_square_logarithmic_error)


acc = metrics.accuracy(target_test, pred_test)
print('Accuracy: {}'.format(acc))


#print_summary(model, line_length=None, positions=None, print_fn=None)