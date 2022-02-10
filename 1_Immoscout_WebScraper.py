# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:04:17 2020

@author: Besitzer
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from random import random
from PIL import Image
from io import BytesIO
from datetime import datetime
from pdf2image import convert_from_path
import os
import ast
import numpy as np

path_data = 'Immoscout-Data/'
path_images = 'Immoscout-Images/'
path_groundplan = 'Immoscout-GroundPlan/'

columns = ['ExposeID', 'City', 'City2', 'City3', 'City4', 'Street', 'HouseNumber', 'ZipCode', 
           'TotalRent', 'BaseRent', 'Area', 'Rooms', 'Bedrooms', 'Bathrooms', 'Floor', 'MaxFloor', 'Title', 
           'Condition', 'Energy', 'EnergyEfficiency', 'Description', 'Equipment', 'Location', 'Miscellaneous', 
           'Balcony', 'Garden', 'Kitchen', 'Cellar', 'Lift', 'AssistedLiving', 'YearBuilt', 'PetsAllowed', 
           'BarrierFree', 'NumberImages', 'ImageText', 'Video', 'GroundPlan']

def web_scraper_immoscout(exposeID, number):
    #set columns for the dataframe
    immo = pd.DataFrame(columns = columns)
    #look for existing ExposeID's to prevent duplicates
    old_immo = pd.DataFrame(columns = immo.columns) 
    for file in os.listdir(path_data):
        im = pd.read_pickle(path_data + file)
        old_immo = pd.concat([old_immo, im]) 
    del im
    old_immo = old_immo.drop_duplicates()
    #set the first URL
    expose = 'https://www.immobilienscout24.de/expose/' + str(exposeID) + '/' 
    for k in range(1, number+2):
        #connect to the URL
        response = requests.get(expose)
        #parse HTML and save to BeautifulSoup object
        soup = BeautifulSoup(response.text, "lxml") #html.parser
        Title = soup.title.text
        match = soup.find('s24-ad-targeting', style='display:none;').text
        dictionary = ast.literal_eval(match[1:])
        #look if this ExposeID is already in the dataset:
        #if this Expose is not in the dataset...
        if (dictionary['obj_scoutId'] not in [exp for exp in immo.ExposeID]) and (dictionary['obj_scoutId'] not in [exp for exp in old_immo.ExposeID]):
            #for every of these columns: empty string if no element found
            if not(soup.find('dd', class_="is24qa-objektzustand grid-item three-fifths")):
                Condition = ''
            else: 
                Condition = soup.find('dd', class_="is24qa-objektzustand grid-item three-fifths").text
            if not(soup.find('dd', class_='is24qa-wesentliche-energietraeger grid-item three-fifths')):
                Energy = ''
            else: 
                Energy = soup.find('dd', class_='is24qa-wesentliche-energietraeger grid-item three-fifths').text
            if not(soup.find('pre', class_='is24qa-objektbeschreibung text-content short-text')):
                Description = ''
            else: 
                Description = soup.find('pre', class_='is24qa-objektbeschreibung text-content short-text').text
            if not(soup.find('pre', class_='is24qa-ausstattung text-content short-text')):
                Equipment = ''
            else:
                Equipment = soup.find('pre', class_='is24qa-ausstattung text-content short-text').text
            if not(soup.find('pre', class_='is24qa-lage text-content short-text')):
                Location = ''
            else: 
                Location = soup.find('pre', class_='is24qa-lage text-content short-text').text
            if not(soup.find('pre', class_='is24qa-sonstiges text-content short-text')):
                Miscellaneous = ''
            else:
                Miscellaneous = soup.find('pre', class_='is24qa-sonstiges text-content short-text').text
            if not(soup.find(class_= 'is24qa-schlafzimmer grid-item three-fifths')):
                Bedrooms = 'Nan'
            else:
                Bedrooms = soup.find(class_= 'is24qa-schlafzimmer grid-item three-fifths').text        
            if not(soup.find(class_="is24qa-badezimmer grid-item three-fifths")):
                Bathrooms = 'Nan'
            else:
                Bathrooms = soup.find(class_="is24qa-badezimmer grid-item three-fifths").text
            #set a list of words and check if these are in the dictionary
            words = ['obj_scoutId', 'obj_regio1', 'obj_regio2', 'obj_regio3', 'geo_krs', 'obj_street', 
                     'obj_houseNumber', 'obj_zipCode', 'obj_totalRent', 'obj_baseRent', 'obj_livingSpace', 'obj_noRooms', 
                     'obj_floor', 'obj_numberOfFloors', 'obj_energyEfficiencyClass', 'obj_balcony', 
                     'obj_garden', 'obj_hasKitchen', 'obj_cellar', 'obj_lift', 'obj_assistedLiving', 'obj_yearConstructed', 
                     'obj_petsAllowed', 'obj_barrierFree', 'obj_picturecount']
            for word in words:
                if not(word in dictionary):
                    dictionary[word] = np.nan
            #set a dataframe for this expose
            obj = pd.DataFrame({'ExposeID'           : [dictionary['obj_scoutId']], 
                                'City'               : [dictionary['obj_regio1']],
                                'City2'              : [dictionary['obj_regio2']],
                                'City3'              : [dictionary['obj_regio3']],
                                'City4'              : [dictionary['geo_krs']],
                                'Street'             : [dictionary['obj_street']],
                                'HouseNumber'        : [dictionary['obj_houseNumber']],
                                'ZipCode'            : [dictionary['obj_zipCode']],
                                'TotalRent'          : [dictionary['obj_totalRent']],
                                'BaseRent'           : [dictionary['obj_baseRent']], 
                                'Area'               : [dictionary['obj_livingSpace']],
                                'Rooms'              : [dictionary['obj_noRooms']],
                                'Bedrooms'           : [Bedrooms],
                                'Bathrooms'          : [Bathrooms],
                                'Floor'              : [dictionary['obj_floor']],
                                'MaxFloor'           : [dictionary['obj_numberOfFloors']],
                                'Title'              : [Title],
                                'Condition'          : [Condition],
                                'Energy'             : [Energy],
                                'EnergyEfficiency'   : [dictionary['obj_energyEfficiencyClass']],
                                'Description'        : [Description],
                                'Equipment'          : [Equipment],
                                'Location'           : [Location],
                                'Miscellaneous'      : [Miscellaneous],
                                'Balcony'            : [dictionary['obj_balcony']],
                                'Garden'             : [dictionary['obj_garden']],
                                'Kitchen'            : [dictionary['obj_hasKitchen']],
                                'Cellar'             : [dictionary['obj_cellar']],
                                'Lift'               : [dictionary['obj_lift']],
                                'AssistedLiving'     : [dictionary['obj_assistedLiving']],
                                'YearBuilt'          : [dictionary['obj_yearConstructed']],
                                'PetsAllowed'        : [dictionary['obj_petsAllowed']],
                                'BarrierFree'        : [dictionary['obj_barrierFree']],
                                'NumberImages'       : [dictionary['obj_picturecount']], 
                                'ImageText'          : '', 
                                'Video'              : '', 
                                'GroundPlan'         : ''})

            #filter all images and the image text except the floorplans
            img_tags = soup.find_all('img')
            image = [img for img in img_tags if 'sp-image' in str(img) and 'floorplan-link' not in str(img)]    
            #save image text
            images_text = ''
            for i in range(len(image)): 
                image_text = str(image[i])
                index = image_text.find('"')
                image_text = image_text[index+1:]
                index = image_text.find('"')
                image_text = image_text[:index]
                images_text = images_text + str(i) + ' ' + image_text + ' '
            obj.ImageText = images_text
            #save images
            images = [img['data-src'] for img in image]
            for i in range(len(images)):
                im = requests.get(images[i])
                img = Image.open(BytesIO(im.content))
                img.save(path_images + obj.ExposeID[0] + '-' + str(i) + ".jpg") 
                #sleep for a couple of seconds
                val = random()
                scaled_value = 1 + (val * 8)
                time.sleep(scaled_value)

            #filter the number of videos
            videolinks = soup.findAll(class_="button slideToVideo is24-fullscreen-gallery-trigger")
            obj.Video = len(videolinks)

            #filter the groundplan as pdf in expose
            groundplan = soup.find(class_="is24-linklist margin-bottom")
            groundplan = str(groundplan)
            obj.GroundPlan = groundplan.count('PDF</span')
            if '" target="' and '<a href="' in groundplan: #just first groundplan
                index = groundplan.find('<a href="')
                groundplan = groundplan[(index+9):]
                index = groundplan.find('" target="')
                groundplan = groundplan[:index]
                im = requests.get(groundplan)
                groundp = convert_from_path(groundplan, 200)
                saveas = path_groundplan + obj.ExposeID[0] + 'GPasPDF' + '.jpg'
                groundp[0].save(saveas, 'JPEG')

            #filter the groundplan as image in expose
            img_tags = soup.find_all('img') #find all images in expose
            floorplan = [img for img in img_tags if 'floorplan-link' in str(img)]
            if len(floorplan) > 0:
                obj.GroundPlan = obj.GroundPlan + len(floorplan)
                floorplan = [img['data-src'] for img in floorplan]
                floorplan = str(floorplan[0]) #just first floorplan
                im = requests.get(floorplan)
                img = Image.open(BytesIO(im.content))
                img.save(path_groundplan + obj.ExposeID[0] + 'GPasImage' + ".jpg") 
            
            # sleep for a couple of seconds
            value = random()
            scaled_value = 1 + (value * 20)
            time.sleep(scaled_value) 

            immo = immo.append(obj,  ignore_index=True)
        
        #if this Expose is already in the dataset...
        else:
            break
            
        #next expose
        links = soup.find_all('a', href=True)
        url = [a['href'] for a in links if 'wohnung-mieten' in str(a)]
        #sorted: the latest expose's first!
        go_to = '/Suche/controller/exposeNavigation/navigate.go?action=NEXT&amp;searchUrl=' + url[0] + '?sorting%3D2&amp;exposeId=' + dictionary['obj_scoutId']
        go_to = go_to.replace('amp;', '')
        expose =  'https://www.immobilienscout24.de' + go_to

        print('Finished Expose-Nr.:', k, '- ExposeID:', dictionary['obj_scoutId'])
    #save the immo-dataframe
    immo.to_pickle(path_data + 
                   str(datetime.now().hour) + '-' +
                   str(datetime.now().minute) + '_' +
                   str(datetime.now().day) + '-' +
                   str(datetime.now().month) + '-' +
                   str(datetime.now().year) + '.pkl') 
    print('End of loop')

web_scraper_immoscout(112352743, 1000)
