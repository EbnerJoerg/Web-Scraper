# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:52:31 2020

@author: Besitzer
"""

import os
import numpy as np
import cv2
import pandas as pd

path = 'C:/Users/Besitzer/Desktop/Seminar_DataRetrieval/'
#manual selection

data = pd.read_pickle(path + 'Berlin_data.pkl')


#list of the image files
kitchen_files = sorted([file for file in os.listdir(path + '/' + 'kitchen_first') if file.endswith('.jpg')])
bathroom_files = sorted([file for file in os.listdir(path + '/' + 'bathroom_first') if file.endswith('.jpg')])
#list of the expose id's
ExposeID_kitchen = [file.split('-')[0] for file in kitchen_files]
ExposeID_bathroom = [file.split('-')[0] for file in bathroom_files]
#all expose id's which are in both, the kitchen_first folder and the bathroom_first folder 
def intersection(lst1, lst2): 
    return [item for item in lst1 if item in lst2]
ExposeID = intersection(ExposeID_kitchen, ExposeID_bathroom)

#create the dataset
dataset = pd.DataFrame(columns=data.columns).transpose()
for ele in ExposeID:
    dataset = pd.concat([dataset, data[data.ExposeID == ele].iloc[0]],axis=1)
dataset = dataset.transpose().reset_index(drop=True)
dataset.to_pickle(path + 'Berlin_kit_bath_first_data.pkl')

