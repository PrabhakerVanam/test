# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:49:24 2020

@author: AD1006362
"""

from dlispy import dump
 # Local File
dlisfilepath = 'D:\\Prabhaker\\Data Science\\Cement Evaluation\\samples\\BB318.dlis'
dlis_output_path= 'D:\\Prabhaker\\Data Science\\Cement Evaluation\\samples\\DLIS output'
## How to dump a dlis file to a readable format
# you can also set eflrOnly as True to only dump the EFLRs as json file load EFLRs
d = dump(dlisfilepath, output_path=dlis_output_path, eflr_only= False) 
  

# load all CSV file from folder
#D:\Prabhaker\Data Science\Cement Evaluation\samples\DLIS output\USI_004PUP
import glob
import numpy as np

directoryPath = 'D:\\Prabhaker\\Data Science\\Cement Evaluation\\samples\\DLIS output\\USI_004PUP\\'
for file_name in glob.glob(directoryPath+'*.csv'):
    x = np.genfromtxt(file_name,delimiter=',')[:,2]
    print(x)

def csv_merge_generator(pattern):
    for file in glob.glob(directoryPath+pattern):
        for line in file:
            yield line

# then using it like this

x = np.genfromtxt(csv_merge_generator('*.csv'))


#dlis = dlisio.load(dlisfilepath)

##How to parse a dlis file and iterate through logical records:

from dlispy import LogicalFile, Object, Attribute, FrameData, PrivateEncryptedEFLR, parse
# you can also set eflrOnly as True to only load EFLRs
_, logical_file_list = parse(dlisfilepath, eflr_only= False) 
for lf  in logical_file_list: # type:LogicalFile
    print("LogicalFile with ID:{}, SequenceNumber:{}".format(lf.id, lf.seqNum))
    for eflr in lf.eflrList:
        if type(eflr) is PrivateEncryptedEFLR:              # PrivateEncryptedEFLR needs to handle separately.
            continue
        print("     Set with Type:{}, Name:{}".format(eflr.setType, eflr.setName))
        for obj in eflr.objects: # type:Object
            print("             Object with Name:{}".format(obj.name))
            for attribute in obj.attributes:    #type:Attribute
                print("                     Attribute with Label:{}, Value:{}, Count:{}, RepCode:{}, Units:{} ".
                      format(attribute.label, ' '.join(map(str, attribute.value)) 
                      if type(attribute.value) is list else attribute.value, 
                      attribute.count, attribute.repCode, attribute.units))

    for frameName, fDataList in lf.frameDataDict.items():
        print("     Frame:{}".format(frameName))
        for fdata in fDataList: # type:FrameData
            print("             FrameData with FrameNumber:{} and {} of slots".
            format(fdata.frameNumber, len(fdata.slots)))
            
            

     

##In this sample code, you will see how to find information about specific object in specific type of EFLR.
from dlispy import LogicalFile, Object, Attribute, FrameData, PrivateEncryptedEFLR, parse
from dlispy import OlrEFLR, FrameEFLR
from dlispy import Frame, FrameData, Origin
# you can also set eflrOnly as True to only load EFLRs
_, logical_file_list = parse(dlisfilepath)
for lf  in logical_file_list: # type:LogicalFile
    for eflr in lf.eflrList:
        if type(eflr) is OlrEFLR:
            for obj in eflr.objects: #type:Origin
                print("File {} created by company {} at {}".format(
                    obj.getAttrValue(Origin.FILE_ID),obj.getAttrValue(Origin.COMPANY),obj.getAttrValue(Origin.CREATION_TIME)))

        if type(eflr) is FrameEFLR:
            for obj in eflr.objects: #type:Frame
                chanel_names = ', '.join(map(str, obj.getAttrValue(Frame.CHANNELS)))
                print("Frame {} with channel list {}".format(obj.name, chanel_names))
                
                
    