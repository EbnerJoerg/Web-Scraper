# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:06:58 2020

@author: Besitzer
"""


import os
import numpy as np
import cv2
import pandas as pd

path = 'Seminar_DataRetrieval/'

#first: manual selection of the kitchen and the bathroom images
data = pd.read_pickle(path + 'Berlin_kit_bath_first_data_prepro.pkl')
data = data.drop_duplicates()
data.ExposeID = data.ExposeID.astype(str)
#list of the image files
kitchen_files = sorted([file for file in os.listdir(path + '/' + 'kitchen_first') if file.endswith('.jpg')])
bathroom_files = sorted([file for file in os.listdir(path + '/' + 'bathroom_first') if file.endswith('.jpg')])
#list of the expose id's
ExposeID_data = data['ExposeID'].drop_duplicates().tolist()
ExposeID_kitchen = [file.split('-')[0] for file in kitchen_files]
ExposeID_bathroom = [file.split('-')[0] for file in bathroom_files]
#all expose id's which are in the dataset, the kitchen_first folder and the bathroom_first folder 
def intersection(lst1, lst2, lst3): 
    return [item for item in lst1 if (item in lst2) and (item in lst3)]
ExposeID = intersection(ExposeID_data, ExposeID_kitchen, ExposeID_bathroom)

#create the final dataset
dataset = pd.DataFrame(columns=data.columns)
for ele in ExposeID:
    dataset = pd.concat([dataset, data[data.ExposeID == ele]])
dataset = dataset.reset_index(drop=True)
dataset.to_pickle(path + 'Berlin_kit_bath_first_data_prepro.pkl')

#create a list with the file names
kit = []
bath = []
for ele in ExposeID:
    k = [file for file in kitchen_files if ele == file.split('-')[0]][0]
    kit.append(k)
    b = [file for file in bathroom_files if ele == file.split('-')[0]][0]
    bath.append(b)

#Image Preprocessing
files = [kit, bath]
category = ['kitchen_first', 'bathroom_first']
def image_preprocessing(width, height):
    for (cat_files, cat) in zip(files, category):
        image_files = sorted([os.path.join(path, cat, file) 
                              for file in cat_files])
        img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in image_files]          
        #resize each images and save them into a numpy array
        dim = (width, height)
        res_img = []      
        for i in range(len(img)):
            res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
            res_img.append(res)
        np.save(path + 'resized_' + cat + '_images.npy', res_img)
image_preprocessing(220,220)


















