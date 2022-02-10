# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:53:21 2020

@author: Besitzer
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
#Tensorflow for CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
#XGBoost for Decision Tree
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import statistics
import matplotlib.dates as mdates
from matplotlib import rcParams


path = "Seminar_DataRetrieval"

kitchen = np.load(os.path.join(path, 'resized_kitchen_first_images.npy'))
bathroom = np.load(os.path.join(path, 'resized_bathroom_first_images.npy'))
data = pd.read_pickle(path + '/' + 'Berlin_kit_bath_first_data_prepro.pkl')

data.Rent_per_Sqm = data.Rent_per_Sqm.replace([np.inf, -np.inf], np.nan)
data.Rent_per_Sqm = data.Rent_per_Sqm.replace(np.nan, data.Rent_per_Sqm.mean())


data = data.drop(['ExposeID'],axis=1)
X_train_kit, X_test_kit, X_train_bath, X_test_bath = train_test_split(kitchen, bathroom, test_size=0.20, random_state=44)
X_train_data, X_test_data, y_train, y_test = train_test_split(data.drop('Rent_per_Sqm', axis=1), data.Rent_per_Sqm, test_size=0.20, random_state=44)
X_train_kit = tf.constant(X_train_kit)
X_test_kit = tf.constant(X_test_kit)
X_train_bath = tf.constant(X_train_bath)
X_test_bath = tf.constant(X_test_bath)
        

#baseline prediction
base_pred = statistics.mean(y_train)
base_pred = pd.Series(base_pred).repeat(len(y_test))
maebase = mean_absolute_error(y_test, base_pred)
msebase = mean_squared_error(y_test, base_pred)
print("several performance metrics baseline:")
print("mae:", maebase)
print("mse:", msebase)


#CNN of the kitchen images
kitchen_image = Input(shape=kitchen.shape[1:4])
#filters, kernel_size (if one integer: height and width is same)
k = Conv2D(32, kernel_size=6, activation='relu')(kitchen_image)
k = MaxPooling2D(pool_size=(3, 3))(k)
k = Conv2D(64, kernel_size=6, activation='relu')(k)
k = MaxPooling2D(pool_size=(3, 3))(k)
k = Conv2D(128, kernel_size=6, activation='relu')(k)
k = MaxPooling2D(pool_size=(3, 3))(k)
k = Flatten()(k)
k = Dense(500, activation='relu')(k)
k = Dropout(0.25)(k)
k = Dense(250, activation='relu')(k)
k = Dropout(0.25)(k)
k = Dense(100, activation='relu')(k)
k = Dropout(0.25)(k)
k = Dense(25, activation='relu')(k)
k = Dense(10, activation="relu")(k)
k = Model(inputs=kitchen_image, outputs=k)

#CNN of the bathroom images
bathroom_image = Input(shape=bathroom.shape[1:4])
#filters, kernel_size (if one integer: height and width is same)
b = Conv2D(32, kernel_size=6, activation='relu')(bathroom_image)
b = MaxPooling2D(pool_size=(3, 3))(b)
b = Conv2D(64, kernel_size=6, activation='relu')(b)
b = MaxPooling2D(pool_size=(3, 3))(b)
b = Conv2D(128, kernel_size=6, activation='relu')(b)
b = MaxPooling2D(pool_size=(3, 3))(b)
b = Flatten()(b)
b = Dense(500, activation='relu')(b)
b = Dropout(0.5)(b)
b = Dense(250, activation='relu')(b)
b = Dropout(0.5)(b)
b = Dense(100, activation='relu')(b)
b = Dropout(0.5)(b)
b = Dense(25, activation='relu')(b)
b = Dense(10, activation="relu")(b)
b = Model(inputs=bathroom_image, outputs=b)

#neural net dataset
input_numeric = Input(shape=(X_train_data.shape[1],))
n = Dense(64, activation="relu")(input_numeric)
n = Model(inputs=input_numeric, outputs=n)



#model kitchen and bathroom image
combined = concatenate([k.output, b.output])
#fully connected layer
z = Dense(10, activation="relu")(combined)
z = Dense(1, activation="linear")(z)
model = Model(inputs=[k.input, b.input], outputs=z)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit([X_train_kit, X_train_bath], y_train, epochs=100, batch_size=128)
prediction = model.predict([X_test_kit, X_test_bath])
maecnn = mean_absolute_error(y_test, prediction)
msecnn = mean_squared_error(y_test, prediction)
print("several performance metrics CNN:")
print("mae:", maecnn)
print("mse:", msecnn)
