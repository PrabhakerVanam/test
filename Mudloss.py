# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:47:47 2020

@author: DELL
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

muddata_loss.set_index(['Report date','Well '], inplace=True, drop = False)

#muddata_loss = muddata_loss.reset_index(level=1, drop=True, inplace=False, col_level=1)

#muddata_loss.drop(columns=['Unnamed: 0'], inplace=True)

print(muddata_loss.shape)

print(muddata_loss.info())

columns_floattype =['Mid Night Depth (ft)','Avg. ROP (ft/hr)','Flow rate (gpm)','Min. RPM (rpm)','Max. RPM (rpm)','Max. WOB (kip)','Min. WOB (kip)','Density (ppg)','PV (cp)','YP (lbf/100ft²)','Mud Loss (bbl)','Subsurface total (bbl)','SPM (spm)','Pressure (psi)']
# Clean the data
muddata_loss[[columns_floattype]] = muddata_loss[[columns_floattype]].replace({'Min. RPM (rpm)': {",": ""}}, regex=True)



muddata_loss= muddata_loss.replace({'Min. RPM (rpm)': {",": ""}}, regex=True)
muddata_loss= muddata_loss.replace({'Max. RPM (rpm)': {",": ""}}, regex=True)
muddata_loss= muddata_loss.replace({'Min. WOB (kip)': {",": ""}}, regex=True)
muddata_loss= muddata_loss.replace({'Max. WOB (kip)': {",": ""}}, regex=True)
muddata_loss= muddata_loss.replace({'TFA (in²)': {",": ""}}, regex=True)
muddata_loss= muddata_loss.replace({'ECD (lbm/ft³)': {",": ""}}, regex=True)
muddata_loss= muddata_loss.replace({'ECD (lbm/ft³)': {",": ""}}, regex=True)

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
    muddata_loss[column] =  muddata_loss[column].astype('float64')
muddata_loss['Min. RPM (rpm)'] =muddata_loss['Min. RPM (rpm)'].astype('float64')
muddata_loss['Max. RPM (rpm)'] =muddata_loss['Max. RPM (rpm)'].astype('float64')
muddata_loss['Min. WOB (kip)'] =muddata_loss['Max. WOB (kip)'].astype('float64')
muddata_loss['Max. WOB (kip)'] =muddata_loss['Max. WOB (kip)'].astype('float64')

print(muddata_loss.isnull().sum())

print(muddata_loss.describe())

print(muddata_loss.dtypes)

print(muddata_loss.head())

