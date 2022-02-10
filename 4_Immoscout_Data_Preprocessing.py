# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:02:51 2020

@author: Besitzer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geocoder


path = 'Seminar_DataRetrieval/'


data = pd.read_pickle(path + 'Berlin_kit_bath_first_data.pkl')


#Location - City
data = data.drop(['City', 'City2', 'City4'], axis=1)
for index in data.index:
    c = data['City3'][index].split('_')
    if len(c) == 4: #Prenzlauer_Berg_Prenzlauer_Berg
        data['City3'][index] = c[0] + ' ' + c[1]
    elif len(c) == 3: #
        if c[1] == c[2]: #Neu_Hohenschönhausen_Hohenschönhausen
            data['City3'][index] = c[0] + ' ' + c[1]
        else: #Französisch_Buchholz_Pankow
            data['City3'][index] = c[0] + ' ' + c[1] + ' ' + c[2]
    elif c[0] == c[1]: #Schöneberg_Schöneberg
        data['City3'][index] = c[0]
    else: #Lichtenrade_Tempelhof
        data['City3'][index] = c[0] + ' ' + c[1]
        
        
#Location - Street, HouseNumber, ZipCode, Longitude, Latitude
#adjustments for the streetnames and housenumbers
street_old = ['Koloniestr.', 'Hilda-Geiringer-Weg', 'Bürgerstr.', 'Schillingstr.',
            'AmGrossenWansee', 'Fischerstr.', 'Joseph-von-Eichendorff-Gasse',
            'AmRuderverein', 'OderbergerStraße', 'Invalidenstraße', 'FrankfurterAllee',
            'UnterdenLinden', 'AnderWuhlheide', 'MetropolitanPark', 'OtelsbergerAllee',
            'AmTegelerHafen', 'Rudolph-von-Gneist-Gasse', 'Tile-Wardenberg-Str.',
            'FalkentalerSteig', 'AmHamburgerBahnhof', 'AmHohenFeld', 'UnterdenEichen', 
            'PeterGastWeg', 'LaehrscherJagdweg', 'Platzdes4Juli', 'BillyWilderPromenade', 'AmBreitenLuch', 'DahlemerWeg', 'StraßezumLöwen', 'kö']
street_new = ['Koloniestraße', 'Hilda-Geiringer-Weg', 'Bürgerstraße', 'Schillingstraße',
            'Am Grossen Wannsee', 'Fischerstraße', 'Joseph-von-Eichendorff-Gasse',
            'Am Ruderverein', 'Oderberger Straße', 'Invalidenstraße', 'Frankfurter Allee',
            'Unter den Linden', 'An der Wuhlheide', 'Metropolitan Park', 'Otelsberger Allee',
            'Am Tegeler Hafen', 'Rudolph-von-Gneist-Gasse', 'Tile-Wardenberg-Straße',
            'Falkentaler Steig', 'Am Hamburger Bahnhof', 'Am Hohen Feld', 'Unter den Eichen',
            'Peter Gast Weg', 'Laehrscher Jagdweg', 'Platz des 4. Juli', 'Billy Wilder Promenade', 'Am Breiten Luch', 'Dahlemer Weg', 'Straße zum Löwen', 'no_information']

#adjustments for the street names
for index in data['Street'].index:
    for ele in range(len(street_old)):
        if street_old[ele] in data['Street'][index]:
            data['Street'][index] = street_new[ele]
#adjustments for the house numbers        
for index in data['HouseNumber'].index:
    data['HouseNumber'][index] = data['HouseNumber'][index].split('/')[0]
    data['HouseNumber'][index] = data['HouseNumber'][index].split('-')[0]
    #if data['HouseNumber'][index] == '0':
    #    data['HouseNumber'][index] = ''
        
key = 'iLJqYuHjlHxN182pRnfUw0I6GyrfVGZ0'
#create columns latitude and longitude
data['Latitude'] = np.nan
data['Longitude'] = np.nan
for i in range(len(data)):
    if pd.isnull(data['Latitude'][i]) or pd.isnull(data['Longitude'][i]):
        if data['Street'][i] == 'no_information':
            Strasse = ''
        else: Strasse = data['Street'][i]
        if data['HouseNumber'][i] == 'no_information' or Strasse == '':
            Hausnummer = ''
        else: Hausnummer = data['HouseNumber'][i]
        if data['ZipCode'][i] == 'no_information':
            PLZ = ''
        else: PLZ = data['ZipCode'][i]
        Stadt = 'Berlin'
        if Strasse == '': #if no street available, then use the district 
            try: #maybe tomtom cannot find the address
                g = geocoder.tomtom(data['City3'][i] + ' ' + PLZ + ' ' + Stadt, key=key)
                data['Latitude'][i] = g.json['lat']
                data['Longitude'][i] = g.json['lng']
            except: print('Address not found')
        else: #street is available
            try: #maybe tomtom cannot find the address
                g = geocoder.tomtom(Strasse + ' ' + Hausnummer+ ', ' + PLZ + ' ' + Stadt, key=key)
                data['Latitude'][i] = g.json['lat']
                data['Longitude'][i] = g.json['lng']
            except: print('Address not found')

data.Latitude[data.Latitude > 53] = np.nan
data.Latitude[data.Latitude < 52] = np.nan
data.Latitude[data.Longitude < 13] = np.nan
data.Latitude[data.Longitude > 14] = np.nan
data.Longitude[data.Latitude.isnull()] = np.nan

#one hot encoding
if 'City3' in data.columns:
    if (data.Longitude.isnull().sum() == 0) and (data.Latitude.isnull().sum() == 0):
        dummyCols = pd.get_dummies(data['City3'])
        for col in dummyCols.columns:
            dummyCols.rename(columns={col: 'District_' + col}, inplace=True)
        data = data.join(dummyCols)
        data = data.drop(['City3', 'Street', 'HouseNumber', 'ZipCode'], axis=1)
        del dummyCols


#Rent - Total, Base, Rent per Squaremeter
data = data.drop('TotalRent', axis=1)
data.BaseRent = data.BaseRent.astype(float)
data.Area = data.Area.astype(float)
data['Rent_per_Sqm'] = data.BaseRent / data.Area
data = data.drop(['BaseRent'], axis=1)
#drop outlier
data = data[data.Rent_per_Sqm < 50].reset_index(drop=True)




#Rooms, Bedrooms, Bathrooms
data.Rooms = data.Rooms.astype(float)
data.Bedrooms = data.Bedrooms.astype(float)
data.Bathrooms = data.Bathrooms.astype(float)
data['Rooms'] = data.Rooms.fillna(data.Rooms.mean()) 
data['Bedrooms'] = data.Bedrooms.fillna(data.Bedrooms.mean()) 
data['Bathrooms'][data.ExposeID == '90140802'] = 1 #manually
data['Bathrooms'] = data.Bathrooms.fillna(data.Bathrooms.mean()) 

data.Floor[data.Floor == 'Nan'] = np.nan
data.MaxFloor[data.MaxFloor == 'Nan'] = np.nan

#Floor, MaxFloor
data.Floor = data.Floor.astype(float)
data.MaxFloor = data.MaxFloor.astype(float)
data['Floor'] = data.Floor.fillna(data.Floor.mean()) 
data['MaxFloor'] = data.MaxFloor.fillna(data.MaxFloor.mean()) 
mean_f = data.Floor.mean()
for i in range(len(data)):
    if data['Floor'][i] > data['MaxFloor'][i]:
        if data['Floor'][i] == mean_f:
            data['Floor'][i] = data['MaxFloor'][i]/2
        else:
            data['MaxFloor'][i] = data['Floor'][i]
            
            
data['Total_Description'] = data['Description'] + ' ' + data['Equipment'] + ' ' + data['Location'] + ' ' + data['Miscellaneous']
#Text - Title, Description, Equipment, Location, Miscellaneous, ImageText
data = data.drop(['Title', 'Description', 'Equipment', 'Location', 'Miscellaneous', 'ImageText'], axis=1)

data['EnergyEfficiency'] = data['EnergyEfficiency'].replace('Nan', np.nan)
#Condition, EnergyEfficiency, PetsAllowed
data['Condition'] = data['Condition'].replace('', np.nan)
data['PetsAllowed'] = data['PetsAllowed'].replace('no_information', np.nan)
column = ['Condition', 'EnergyEfficiency', 'PetsAllowed']
for word in column:
    if word in data.columns:
        dummyCols = pd.get_dummies(data[word])
        for col in dummyCols.columns:
            dummyCols.rename(columns={col: word + '_' + col}, inplace=True)
        data = data.join(dummyCols)
        del data[word], dummyCols
del word, column



#data['Energy'].value_counts()
#Energy
energy_cat = ['Fernwärme', 'Gas', 'Öl', 'KWK fossil', 'Erdgas leicht', 'KWK erneuerbar',
              'Erdwärme', 'Erdgas schwer', 'Fernwärme-Dampf', 'Holzpellets', 'Nahwärme',
              'Strom', 'Umweltwärme', 'Bioenergie', 'KWK regenerativ', 'Fernwärme', 
              'Wärmelieferung', 'Kohle', 'Solar', 'Kohle']
#create new columns
for ele in energy_cat:
    data['Energy_' + ele] = np.nan
data['Energy'] = data['Energy'].replace('', np.nan)
for row in range(len(data)):
    for ele in energy_cat:
        if row in data[data.Energy.isnull()].index:  
            data['Energy_' + ele][row] = 0
        else: 
            if ele in data['Energy'][row]:
                data['Energy_' + ele][row] = 1
            else: data['Energy_' + ele][row] = 0
data = data.drop('Energy', axis=1)

#Balcony, Garden, Kitchen, Cellar, Lift, BarrierFree
#only yes or no --> yes = 1, no = 0
liste = ['Balcony', 'Garden', 'Kitchen', 'Cellar', 'Lift', 'BarrierFree']
for element in liste:
    data[element] = data[element].replace('y', 1).replace('n', 0)

#AssistedLiving
data = data.drop('AssistedLiving', axis=1)

data.YearBuilt[data.YearBuilt == 'Nan'] = np.nan
#YearBuilt
#create new column Age
data.YearBuilt = data.YearBuilt.astype(float)
data['Age'] = 2020 - data['YearBuilt']
data = data.drop('YearBuilt', axis=1)
data.Age = data.Age.fillna(data.Age.mean())

#change the type of some columns
for col in data.columns:
    if (data[col].dtypes == 'object'):
        data[col] = data[col].astype(int)


data.to_pickle(path + 'Berlin_kit_bath_first_data_prepro.pkl')
