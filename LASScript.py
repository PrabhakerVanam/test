# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:49:24 2020

@author: AD1006362
"""

import lasio
import laspy
from dlispy import dump
import dlisio
import petropy

 

import matplotlib.pyplot as plt

 

las = lasio.read(r'D:\\Prabhaker\\Data Science\\Cement Evaluation\\samples\\REP6124 SA-59_metric.las')
las.curves
df=las.df()
df.head()
df.shape
df.columns
 

df_selected = df[['GR', 'NEUT']]
 

df_selected.isna().sum()


df_dropped = df_selected.dropna(subset=['GR', 'NEUT'],axis=0, how='any')


df_dropped.describe()

 

#df_filt = df_dropped[(df_dropped.GR > 0) & (df_dropped.GR  <= 250)]

 

df_idx = df.rename_axis('Depth').reset_index()
df_idx.head()

 

 


def log_plot(logs):
    logs = logs.sort_values(by='Depth')
    top = logs.Depth.min()
    bot = logs.Depth.max()
    
    f, ax = plt.subplots(nrows=1, ncols=3,figsize=(10,8))
    ax[0].plot(logs.GR, logs.Depth, color='green')
    ax[1].plot(logs.NEUT, logs.Depth, color='red')

 

    
    for i in range(len(ax)):
        ax[i].set_ylim(top,bot)
        ax[i].invert_yaxis()
        ax[i].grid()
        
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[0].set_ylabel("Depth(ft)")
    ax[1].set_xlabel("NEUT")
    ax[1].set_xlim(logs.NEUT.min(),logs.NEUT.max())

 

    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]);
 
    f.suptitle('Well:drake', fontsize=14,y=0.94)

 


log_plot(df_idx)