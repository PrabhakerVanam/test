# -*- coding: utf-8 -*-
"""
Created on Sat May 2 18:36:24 2020

@author: AD1006362
"""

import matplotlib.pyplot as plt
 
import pandas as pd

import numpy as np

import dlisio


import os

import sys 

from dlispy import dump

import time

import logging

asset_well_name = "ASAB"

vendor_name = "BAKER"

dlis_file_name="BB318.dlis"

baseFolder = "D:\\Prabhaker\\Data Science\\Cement Evaluation\\{}\\{}".format(vendor_name, asset_well_name)

outputFolder ="{}\\{}".format(baseFolder, "Output")

# Create Base folder if it doesnt exists
try:
    os.mkdir(outputFolder)
    print("Directory " , outputFolder ,  " Created ") 
except FileExistsError:
    print("Directory " , outputFolder ,  " already exists")


source_dlis_file = "{}\\{}\\{}".format(baseFolder, "Input", dlis_file_name)

print("Input dlis file : " + source_dlis_file)
#322754842_BB-1459-1-1H_BB1459101H_20200206_0700_0958_SBT_1200PSI.dlis
#308523172_BB-1013-1-1H_BB1013101H_20140727_0700_0958_SBT_DLIS.dlis

producer_name=""

namespace_name = ""

well_name=""

field_name=""

file_number=""


def paint_channel(ax, curve, y_axis, x_axis, **kwargs):
    """Plot an image channel into an axes using an index channel for the y-axis
    
    Parameters
    ----------
    
        ax : matplotlib.axes
        
        curve : numpy array
            The curve to be plotted
        
        index : numpy array 
            The depth index as a Channel object (slower) or a numpy array (faster)
        
        **kwargs : dict 
            Keyword arguments to be passed on to ax.imshow()
    """
    # Determine the extent of the image so that the pixel centres correspond with the correct axis values
    dx = np.mean(x_axis[1:] - x_axis[:-1])
    dy = np.mean(y_axis[1:] - y_axis[:-1])
    extent = (x_axis[0] - dx/2, x_axis[-1] + dx/2, y_axis[0] - dy/2, y_axis[-1] + dy/2)
    
    # Determine the correct orientation of the image
    if y_axis[1] < y_axis[0]:   # Frame recorded from the bottom to the top of the well
        origin = 'lower'
    else:                       # Frame recorded from the top to the bottom of the well
        origin = 'upper'
    
    return ax.imshow(curve, aspect='auto', origin=origin, extent=extent, **kwargs)

def index_of(frame):
    """Return the index channel of the frame"""
    return next(ch for ch in frame.channels if ch.name == frame.index)

def get_channel(frame, name):
    """Get a channel with a given name from a given frame; fail if the frame does not have exactly one such channel"""
    [channel] = [x for x in frame.channels if x.name == name]
    return channel

def plot_wave_form(frame, curvedata, channel_names):
    
    wf1 = get_channel(frame, channel_names[0] )
    
    wf2 = get_channel(frame, channel_names[1] )
    
    # Determine the maximum absolute value of WF1 and WF2 so that we can balance the colormap around 0
    wf_max = max(np.max(np.abs(curvedata[channel_names[0]])), np.max(np.abs(curvedata[channel_names[1]])))
    wf_lim = 0.5 * wf_max
    
    # Parameters for plotting the waveforms
    wf_pltargs = {
        'cmap': 'seismic',
        'vmin': -wf_lim,
        'vmax': wf_lim,
    }
    
    wf_samples = np.arange(wf1.dimension[0])   # x values to use in plotting
    
    # Create figure and axes
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 12), constrained_layout=True)
    
    # Plot WF1 as an image
    ax = axes[0]
    im = paint_channel(ax, curvedata[channel_names[0]], curvedata[frame.index], wf_samples, **wf_pltargs)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label(f'{wf1.name}: {wf1.long_name}')
    ax.set_ylabel('Depth $z$ [m]')
    
    # Plot WF1 as an image
    ax = axes[1]
    im = paint_channel(ax, curvedata[channel_names[1]], curvedata[frame.index], wf_samples, **wf_pltargs)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label(f'{wf2.name}: {wf2.long_name}')
    ax.set_ylabel('Depth $z$ [m]')
    
    for ax in axes:
        ax.set_xlabel('Sample $k$')
        ax.grid(True)
        

def plot_single_wave_form(frame, curvedata, channel_name):
    
    wf1 = get_channel(frame, channel_name )
    
    # Determine the maximum absolute value of WF1 and WF2 so that we can balance the colormap around 0
    wf_max = np.max(np.abs(curvedata[channel_name]))
    wf_lim = 0.5 * wf_max
    
    # Parameters for plotting the waveforms
    wf_pltargs = {
        'cmap': 'seismic',
        'vmin': -wf_lim,
        'vmax': wf_lim,
    }
    
    wf_samples = np.arange(wf1.dimension[0])   # x values to use in plotting
    
    # Create figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(9, 12), constrained_layout=True)
    
    # Plot WF1 as an image
    ax = axes
    im = paint_channel(ax, curvedata[channel_name], curvedata[frame.index], wf_samples, **wf_pltargs)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label(f'{wf1.name}: {wf1.long_name}')
    ax.set_ylabel('Depth $z$ [m]')
    ax.set_xlabel('Sample $k$')



f, *f_tail = dlisio.load(source_dlis_file)

if len(f_tail): logging.warning('There are more logical files in tail')

origin, *origin_tail = f.origins

if len(origin_tail): logging.warning('f contains multiple origins')

origin.describe()

f.describe()


for frame in f.frames:
    index_channel = next(ch for ch in frame.channels if ch.name == frame.index)
    print(f'Frame {frame.name}:')
    print(f'Description      : {frame.description}')
    print(f'Indexed by       : {frame.index_type}')
    print(f'Interval         : [{frame.index_min}, {frame.index_max}] {index_channel.units}')
    print(f'Direction        : {frame.direction}')
    print(f'Constant spacing : {frame.spacing} {index_channel.units}')
    print(f'Index channel    : {index_channel}')
    print(f'No. of channels  : {len(frame.channels)}')
    print()

# Get 10B and its index channel
frame60B = f.object('frame', '60B')
index60B = index_of(frame60B)
 
curves60B = frame60B.curves()
 
# Convert the index to metres if needed
if index60B.units == '0.1 in':
    curves60B[frame60B.index] *= 0.00254
 
print('Shape of depth index curve array:', curves60B[frame60B.index].shape)
print('Shape of WAVE curve array:        ', curves60B['U001'].shape)

wf_column_list=['U010','U011']
# Plot wave data with multi columns
plot_wave_form(frame60B, curves60B,wf_column_list)
# Plot wave data with single column
plot_single_wave_form(frame60B, curves60B,'U012')
      
