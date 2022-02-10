# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import shutil, os

#input and output path
path_data = 'Immoscout-Data/'
path_images = 'Immoscout-Images/'
path_data_dest = 'Seminar_DataRetrieval/'

#columns
columns = ['ExposeID', 'City', 'City2', 'City3', 'City4', 'Street', 'HouseNumber', 'ZipCode', 
            'TotalRent', 'BaseRent', 'Area', 'Rooms', 'Bedrooms', 'Bathrooms', 'Floor', 'MaxFloor', 'Title', 
            'Condition', 'Energy', 'EnergyEfficiency', 'Description', 'Equipment', 'Location', 'Miscellaneous', 
            'Balcony', 'Garden', 'Kitchen', 'Cellar', 'Lift', 'AssistedLiving', 'YearBuilt', 'PetsAllowed', 
            'BarrierFree', 'NumberImages', 'ImageText', 'Video', 'GroundPlan']

#scraped data
old_immo = pd.DataFrame(columns = columns) 
for file in os.listdir(path_data):
    im = pd.read_pickle(path_data + file)
    old_immo = pd.concat([old_immo, im]) 
del im
old_immo = old_immo.drop_duplicates()
#select the dataset of Berlin and save it
Berlin = old_immo[old_immo.City == 'Berlin']
Berlin = Berlin.reset_index(drop=True)
Berlin.to_pickle(path_data_dest + 'Berlin_data.pkl')

#create a dataframe with the images
Berlin['NumberImages'] = Berlin['NumberImages'].astype(int)
name_list_df = []
for row in range(len(Berlin)): #each row
    #saves the expose id
    name_list = [Berlin['ExposeID'][row]]
    file = Berlin['ImageText'][row]
    for image in range(Berlin['NumberImages'][row]+1):
        if image == 0: #removes the number in front of the first image
            file = file.split(str(image) + ' ', 1)[1:]
            file = ' '.join(file)
        if image != 0: #saves the image name and removes the next number
            name = file.split(' ' + str(image) + ' ')[0]
            file = file.split(str(image) + ' ', 1)[1:]
            file = ' '.join(file)
            name_list.append(name)
    name_list_df.append(name_list)
name_images = pd.DataFrame(name_list_df)
name_images.rename(columns={'0':'ExposeID'}, inplace=True)
name_images.to_pickle(path_data_dest + 'image_names.pkl')

#key words and the list of the categories
kitchen = ['kitchen', 'Küche', 'küche', 'Koch', 'koch']
bathroom = ['bathroom', 'Bad', 'bad', 'WC', 'wc', 'Toilette', 'toilette', 'Dusche','dusche', 'Klo', 'klo']
liste = [kitchen, bathroom]
liste_words = []
for word in liste:
    for ele in word:
        liste_words.append(ele)
        
for row in range(len(name_images)):
    #now look if this row contains some key words
    res = len([ele for ele in liste_words if(ele in Berlin['ImageText'][row])]) != 0
    if res == True:
        ExposeID = name_images.loc[row,0]
        length = Berlin['NumberImages'][row]
        #select category
        for cat in liste:
            i = 0 #to only take the first kitchen/bathroom image
            #going through each column
            for col in range(1, length+1):
                name = name_images.loc[row,col]
                # if cell contains a key word and it is the first cell of this column
                if (len([word for word in cat if(word in name)]) > 0) and (i == 0):
                    filename = ExposeID + '-' + str(col-1) + '.jpg'
                    i = i + 1
                    try:
                        shutil.copy(path_images + filename, 
                        path_data_dest + cat[0] + '_first' + '/' + filename)
                    except FileNotFoundError: #should be no error, only if not all images are in the path
                        print('no such file or directory')







