#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:56:48 2024

@author: samuelpelaez
"""
#%%
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


#%%
# Actualizar la función para asignar la parte del día con la nueva división horaria
def updated_assign_part_of_day(std_time):
    hour = pd.to_datetime(std_time).hour
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'
    
#%%

#dataTrain = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Training Set Normalizado.csv')
dataTrain = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Training Set No Normalizado.csv')

dataTrain = dataTrain.drop('Unnamed: 0', axis = 1)

scaler = MinMaxScaler()

# Convertir la serie a DataFrame y aplicar fit_transform
dataTrain['FlightTime'] = scaler.fit_transform(dataTrain[['FlightTime']])
product_columns = dataTrain.columns.difference(['FlightTime', 'Aforo', 'DepartureStation','ArrivalStation', 'PartOfDay', 'Month', 'Day', 'Weekday'])

#%%

# Features (X): todas las columnas excepto las de productos
X = dataTrain.drop(product_columns, axis=1)

# Targets (Y): solo las columnas de productos
Y = dataTrain[product_columns]

#%%

model1 = Sequential()
model1.add(Dense(32, input_dim=X.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(Y.shape[1], activation='linear'))  # Ajusta según el número de productos a predecir

model1.compile(optimizer='adam', loss='mse')

# Configuración de Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Entrenamiento del modelo
model1.fit(X, Y, epochs=100, batch_size=10, validation_split=0.2, callbacks=[early_stopping])

#%%

dataPredict = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Preds2.csv')

dataPredict = dataPredict.drop('Unnamed: 0', axis = 1)

for i in product_columns:
    dataPredict[i] = ''
    
dataPredict['STA'] = pd.to_datetime(dataPredict['STA'])
dataPredict['STD'] = pd.to_datetime(dataPredict['STD'])
dataPredict['FlightTime'] = dataPredict['STA']-dataPredict['STD']
dataPredict['Aforo'] = dataPredict['Passengers']/dataPredict['Capacity']
dataPredict.loc[dataPredict['Aforo'] > 1, 'Aforo'] = 1
dataPredict['PartOfDay'] = dataPredict['STD'].apply(updated_assign_part_of_day)
dataPredict['Month'] = pd.to_datetime(dataPredict['STD']).dt.month
dataPredict['Day'] = pd.to_datetime(dataPredict['STD']).dt.day
dataPredict['Weekday'] = pd.to_datetime(dataPredict['STD']).dt.weekday
dataPredict['FlightTime'] = pd.to_timedelta(dataPredict['FlightTime']).dt.total_seconds()

# Convertir 'FlightTime' a minutos desde segundos
dataPredict['FlightTime'] = dataPredict['FlightTime'] / 60

dataPredict['FlightTime'] = scaler.fit_transform(dataPredict[['FlightTime']])


predict_columns = ['FlightTime', 'Aforo', 'DepartureStation','ArrivalStation', 'PartOfDay', 'Month', 'Day', 'Weekday']

assert all(col in dataPredict.columns for col in predict_columns), "Some columns are missing in the DataFrame!"


# Obtener las columnas que no están en columns_to_move
remaining_columns = [col for col in dataPredict.columns if col not in predict_columns]

# Nueva lista de columnas, poniendo 'columns_to_move' al final
new_column_order = remaining_columns + predict_columns

# Reordenar el DataFrame según la nueva lista de columnas
df1 = dataPredict[new_column_order]

#%%$

le = LabelEncoder()

dataPredict['PartOfDay'] = le.fit_transform(dataPredict['PartOfDay'])
dataPredict['DepartureStation' ] = le.fit_transform(dataPredict['DepartureStation'])
dataPredict['ArrivalStation'] = le.fit_transform(dataPredict['ArrivalStation'])

#%%

XP = dataPredict[['FlightTime', 'Aforo', 'DepartureStation', 'ArrivalStation', 'PartOfDay', 'Month', 'Day', 'Weekday']]

# Diccionario para almacenar las predicciones
predictions = model1.predict(XP)

for i, col in enumerate(product_columns):
    dataPredict[col] = predictions[:, i]

#%%

# Redondear valores y convertir negativos a 0
for column in product_columns:
    dataPredict[column] = dataPredict[column].clip(lower=0).round(0)
    
#%%

dataPredict['Productos Totales'] = dataPredict[product_columns].sum(axis=1)

#%%

dataPredict.to_csv('Predicciones Productos NN.csv')