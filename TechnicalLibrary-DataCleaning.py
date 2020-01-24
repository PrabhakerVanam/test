# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:50:41 2020

@author: DELL
"""

import pandas as pd

df = pd.read_excel('D:\\Prabhaker\\Data Science\\Keras\\TechnicalLibrary-Metadata.xlsx')

df.head()

#page_df= df.iloc[1:200,:]
page_df= df.iloc[:,:]

page_df.columns=['Value']
page_df.describe()
print(page_df.dtypes)
page_df['Value'] = page_df['Value'].astype('str')
page_df['Key'] = page_df['Value'].apply(lambda x: 3 if x.find('Copy: 1') > -1 else (4 if x.find('ITEM ID:') > -1 else (5 if x.find('Local Call Number:')  > -1 else ( 6 if x.find('Date Received:') > -1  else 999))) )


page_df.reset_index(drop=True, inplace=True)

# =============================================================================
# page_df.iloc[18,1]
# page_df.iloc[19,1]
# page_df.iloc[20,1]
# 999==999
# =============================================================================
page_df.iloc[10,0]

page_df_new = page_df.copy()
# drows rows where you have Page x of 5837
for i in range(0, len(page_df.index)):         
    if (page_df.iloc[i,0].find("of 5837") > -1): 
        #print(page_df.iloc[i,0])
        page_df_new.drop(index=i, axis=0, inplace=True)
        
page_df_new.reset_index(drop=True, inplace=True)

page_df = page_df_new.copy()
#drows rows where you have 2020-01-12 OR 
1/12/2020
for i in range(0, len(page_df.index)):         
    if (page_df.iloc[i,0].find("2020-01-12") > -1):         
        #print(page_df.iloc[i,0])
        page_df_new.drop(index=i, axis=0, inplace=True)
    elif(page_df.iloc[i,0].find("
1/12/2020") > -1):
        page_df_new.drop(index=i, axis=0, inplace=True)

page_df_new.reset_index(drop=True, inplace=True)
# copy the updated dataframe back to original
page_df = page_df_new.copy()


# iterate throug all indexes and merge subject what it is splitted into two rows
for i in range(1, len(page_df.index)-1):         
    if ((page_df.iloc[i-1,1] == page_df.iloc[i,1])):        
        if(page_df.iloc[i,1] == page_df.iloc[i+1,1] ): 
            page_df.iloc[i,0] = page_df.iloc[i,0] + " "+ page_df.iloc[i+1,0]
           
page_df_new = page_df.copy()
# drop the row where subject splitted in two rows
for i in range(1, len(page_df.index)-1):         
    if ((page_df.iloc[i-1,1] == page_df.iloc[i,1])):        
        if(page_df.iloc[i,1] == page_df.iloc[i+1,1] ): 
            page_df_new.drop(index=i, axis=0, inplace=True)

page_df_new.reset_index(drop=True, inplace=True)
            
page_df = page_df_new.copy()
#reset the index for the Author and Subject  based on two rows with same key 999
for i in range(0, len(page_df.index)-1):         
    if ((page_df.iloc[i,1] == page_df.iloc[i+1,1])):  
        page_df_new.iloc[i,1]=1
        page_df_new.iloc[i+1,1]=2
        
page_df_new.reset_index(drop=True, inplace=True)
            
page_df = page_df_new.copy()

dict = {1:'Author', 2:'Subject',3:'Copy_No', 4:'Item_Id', 5:'Report_No', 6:'Received_Date'}
page_df_new['Attribute'] = page_df_new['Key'].replace(dict)


#page_df_new.pivot(index='Key', columns='Attribute', values='Attribute')