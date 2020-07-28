# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:44:21 2020

@author: Besitzer
"""
#plots
import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle

##Change fontsize
matplotlib.rcParams.update({'font.size': 14})
## Change font
matplotlib.rcParams["font.family"] = "cmr10"


path = 'C:/Users/Besitzer/Desktop/Seminar_DataRetrieval/'

#Missing Values
data = pd.read_pickle(path + 'Berlin_kit_bath_first_data.pkl')
no_inf = ['Street', 'HouseNumber', 'ZipCode', 'PetsAllowed']
for ele in no_inf:
    data[ele][data[ele] =='no_information'] = np.nan
data['EnergyEfficiency'][data['EnergyEfficiency'] =='NO_INFORMATION'] = np.nan
Nan = ['Rooms', 'Bedrooms', 'Bathrooms', 'Floor', 'MaxFloor', 'YearBuilt']
for ele in Nan:
    data[ele][data[ele] =='Nan'] = np.nan   
text = ['Title', 'Condition', 'Description', 'Equipment', 'Location', 'Miscellaneous', 'ImageText', 'Energy']
for ele in text:
    data[ele][data[ele] ==''] = np.nan   
fig = msno.matrix(data)
plt.title('Missing Values', size=40)
fig_copy = fig.get_figure()
#plt.tight_layout()
fig_copy.savefig(path + 'Seminar_Paper/Plots/missing_values.png',bbox_inches='tight')


#Boxplot
matplotlib.rcParams.update({'font.size': 12})
#data.to_pickle(path + 'Berlin_boxplot_outliers.pkl')
data = pd.read_pickle(path + 'Berlin_boxplot_outliers.pkl')
data.BaseRent = data.BaseRent.astype(float)
data.Area = data.Area.astype(float)
data['Rent_per_Sqm'] = data.BaseRent / data.Area
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Boxplot Rent per Square Meter', y = 0.94)
plt.yticks([], [''])
ax1.set_yticks([]) 
ax1.set_yticklabels([''])
ax1.set_ylabel('with\noutliers', rotation=0)
ax1.yaxis.set_label_coords(-0.07, 0.38)
ax2.set_ylabel('without\noutliers', rotation=0)
ax2.yaxis.set_label_coords(-0.07, 0.38)
ax1.boxplot(data.Rent_per_Sqm, vert=False)
ax2.boxplot(data.Rent_per_Sqm[data.Rent_per_Sqm < 60], vert=False)
plt.savefig(path + 'Seminar_Paper/Plots/boxplot_outliers.pdf')



#Scatterplot - Lat Long
#data.to_pickle(path + 'Berlin_scatter_latlong.pkl')
data = pd.read_pickle(path + 'Berlin_scatter_latlong.pkl')
matplotlib.rcParams.update({'font.size': 10})
fig, ax = plt.subplots()
points = plt.scatter(x='Longitude', y='Latitude', data=data,
                     c='Rent_per_Sqm', s=3, cmap="Blues")
plt.title('Scatterplot Rent per Square Meter')
z = [13.3800969,13.3948593,13.3653063,13.2288927,13.1256896,13.1599921,
     13.3037243,13.3917517,13.5303404,13.5176208,13.4419289,13.2251513]
y = [52.522273,52.5074159,52.5977691,52.5080245,52.5192793,52.4296191,
     52.4406917,52.4459959,52.417978,52.5225623,52.5323095,52.6048723]
n = ['Mitte', 'Friedrichshain-Kreuzberg', 'Pankow',
     'Charlottenburg-\nWilmersdorf', 'Spandau', 'Steglitz-\nZehlendorf',
     'Tempelhof-\nSchoeneberg', 'Neukoelln','Treptow-Koepenick',
     'Marzahn-Hellersdorf','Lichtenberg','Reinickendorf']
#ax.scatter(z, y)
for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]), size=6, ha='center', va='center')
plt.colorbar(points)
sns.regplot(x='Longitude', y='Latitude', data=data, scatter=False, fit_reg=False,  color=".1")
plt.savefig(path + 'Seminar_Paper/Plots/scatter_latlong.pdf')



#Hist of Prediction
#prediction of kit bath
pred_data = pd.read_pickle(path + 'prediction_data.pkl')
matplotlib.rcParams.update({'font.size': 10})
fig, ax = plt.subplots()
plt.hist(pred_data.Absolute_Error, bins=21, density=1,
         edgecolor='steelblue', linewidth=.6, color='skyblue')
plt.xlabel('Absolute Error')
ax.set_ylabel('Relative\nFrequency', rotation=0)
ax.yaxis.set_label_coords(-0.14, 0.4)
plt.title('Histogram of Absolute Prediction Error')
plt.tight_layout()
plt.savefig(path + 'Seminar_Paper/Plots/Hist_Predictions.pdf')
#komplett falsch: testset nummer 159
#iloc von 1259
#expose nummer: 117147445



#Model_Training_History
pickle_in = open(path + 'HistoryDictModel1','rb')
HistoryDictModel1 = pickle.load(pickle_in)
m1 = HistoryDictModel1['val_loss']
pickle_in = open(path + 'HistoryDictModel2','rb')
HistoryDictModel2 = pickle.load(pickle_in)
m2 = HistoryDictModel2['val_loss']
pickle_in = open(path + 'HistoryDictModel3','rb')
HistoryDictModel3 = pickle.load(pickle_in)
m3 = HistoryDictModel3['val_loss']
pickle_in = open(path + 'HistoryDictModel4','rb')
HistoryDictModel4 = pickle.load(pickle_in)
m4_rmse = HistoryDictModel4['validation_1']['rmse']
m4 = [ele**2 for ele in m4_rmse]
pickle_in = open(path + 'HistoryDictModel5','rb')
HistoryDictModel5 = pickle.load(pickle_in)
m5_rmse = HistoryDictModel5['validation_1']['rmse']
m5 = [ele**2 for ele in m5_rmse]
file = 'Model_Training_History'
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams["font.family"] = "cmr10"
#Values
#x = [1,2,3,4,5,6,7,8,9,10]
#y = x
fig,ax = plt.subplots(1,1)#, figsize=(10,5))
#x_pos = [2,6,10]
#plt.xticks(x_pos, (2,6,10))
#y_pos = [2,6,10]
#plt.yticks(y_pos, (2,6,10))
#Set_label:
#rotation: 0 (horizontal), default is 90, labelpad: space to axis
ax.set_ylabel('Mean\nSquared\nError', rotation=0, labelpad=30)
ax.yaxis.set_label_coords(-0.09, 0.425)
ax.set_xlabel('Epoch', labelpad=0)
ax.set_ylim(0,50)
ax.set_xlim(0,200)
plt.plot(m1, 'teal')
plt.plot(m2, 'skyblue')
plt.plot(m3, 'steelblue')
plt.plot(m4, 'forestgreen')
plt.plot(m5, 'darkseagreen')
plt.hlines(34.227, 0, 200, linestyles='dotted')
plt.title('Mean Squared Error for several Models')
#plt.plot(x,y, label='example graph')
#Legend:
#loc: best, 2 of: upper/lower + center + left/right, right, center
#bbox_to_anchor: (x, y, width, height) or (x, y)
#markerscale: default is None --> 1
plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Baseline'], loc="upper right",  fontsize=11.5)#, bbox_to_anchor=(0.98, 0.98))
plt.tight_layout()
plt.savefig(path + '/Seminar_Paper/Plots/' + file + '.pdf')

'skyblue', 'steelblue', 'forestgreen', 'orangered', 'darkseagreen',
'brown', 'darkslategray'




#Model1_Training_History
pickle_in = open(path + 'HistoryDictModel1','rb')
HistoryDictModel1 = pickle.load(pickle_in)
m = HistoryDictModel1['loss']
m1 = HistoryDictModel1['val_loss']
file = 'Model1_Training_History'
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams["font.family"] = "cmr10"
fig,ax = plt.subplots(1,1)#, figsize=(10,5))
ax.set_ylabel('Mean\nSquared\nError', rotation=0, labelpad=30)
ax.yaxis.set_label_coords(-0.09, 0.425)
ax.set_xlabel('Epoch', labelpad=0)
ax.set_ylim(0,50)
ax.set_xlim(0,200)
plt.plot(m, 'steelblue')
plt.plot(m1, 'teal')
plt.hlines(34.227, 0, 200, linestyles='dotted')
plt.title('Mean Squared Error for Model 1')
#plt.plot(x,y, label='example graph')
#Legend:
#loc: best, 2 of: upper/lower + center + left/right, right, center
#bbox_to_anchor: (x, y, width, height) or (x, y)
#markerscale: default is None --> 1
plt.legend(['Train Error', 'Test Error', 'Baseline'], loc="upper right",  fontsize=11.5)#, bbox_to_anchor=(0.99, 0.99))
plt.tight_layout()
plt.savefig(path + '/Seminar_Paper/Plots/' + file + '.pdf')



