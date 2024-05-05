#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 11:45:35 2024

@author: samuelpelaez
"""
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow
from sklearn.preprocessing import MinMaxScaler


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
#Lectura de files

sales = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Sales TEC_Valid (1).csv')
flights = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Filghts TEC_Valid.csv')

#%%

#Separar por años
flights['STD'] = pd.to_datetime(flights['STD'])
flights['STA'] = pd.to_datetime(flights['STA'])


flights23 = flights[(flights['STD'].dt.year == 2023) & (flights['STA'].dt.year == 2023)]
fligths2425 = flights[(flights['STD'].dt.year >= 2024) | (flights['STA'].dt.year >= 2024)]

#%%

#Generar tiempod e vuelo

flights23['FlightTime'] = flights23['STA']-flights23['STD']

#%%

#Generar tabla de productos

sales['UnitPrice'] = sales['TotalSales'] / sales['Quantity']
productos = sales[['ProductType', 'ProductName','UnitPrice']].drop_duplicates()
productos1 = sales[['ProductName']].drop_duplicates()

#%%

#Tabla de rutas

flights['Rutas'] = flights['DepartureStation']+'-'+flights['ArrivalStation']
flights23['Aforo'] = flights23['Passengers']/flights23['Capacity']
rutas = flights[['DepartureStation', 'ArrivalStation']].drop_duplicates()


#%%

#Tabla de productos únicos con productos

# Calcular el precio unitario si aún no está calculado
sales['UnitPrice'] = sales['TotalSales'] / sales['Quantity']


#%%

#Normalización

scaler = StandardScaler()

# Normalizar la columna 'TotalSales'
sales['TotalSales_Standardized'] = scaler.fit_transform(sales[['TotalSales']])

valores_a_eliminar = ['COMBOS CREW',
'Hertz.', 'OFERTAS ', 'Transportaciones MTY',
'Transportaciones TLC', 'VIVA PLAY', 'VIVA Taxis', 'Antros',
'VivaTransfer','Vivabus', 'Transportaciones CUN','Specials']

sales1 = sales[~sales['ProductType'].isin(valores_a_eliminar)]

#%%
# Suponiendo que sales y flights23 ya están cargados

# Crear la tabla pivot
productsPerFlight = sales1.pivot_table(
    index='Flight_ID',
    columns='ProductName',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
).reset_index()  # Aquí reseteamos el índice para hacer 'Flight_ID' una columna regular

productsPerFlight = productsPerFlight.drop_duplicates(subset=['Flight_ID'])
productsPerFlight = pd.merge(productsPerFlight, flights23[['Flight_ID', 'FlightTime']], on='Flight_ID', how='left')
productsPerFlight = pd.merge(productsPerFlight, flights23[['Flight_ID', 'STD']], on='Flight_ID', how='left')
productsPerFlight = pd.merge(productsPerFlight, flights23[['Flight_ID', 'Aforo']], on='Flight_ID', how='left')
productsPerFlight = pd.merge(productsPerFlight, flights23[['Flight_ID', 'DepartureStation']], on='Flight_ID', how='left')
productsPerFlight = pd.merge(productsPerFlight, flights23[['Flight_ID', 'ArrivalStation']], on='Flight_ID', how='left')
productsPerFlight.loc[productsPerFlight['Aforo'] > 1, 'Aforo'] = 1

#%%
le = LabelEncoder()

productsFlightsAnalysis = productsPerFlight.copy()

productsFlightsAnalysis = productsFlightsAnalysis.dropna()
productsFlightsAnalysis = productsFlightsAnalysis.drop('Flight_ID', axis = 1)

# Aplicar la función actualizada para crear una nueva columna 'PartOfDay'
productsFlightsAnalysis['PartOfDay'] = productsFlightsAnalysis['STD'].apply(updated_assign_part_of_day)

productsFlightsAnalysis['FlightTime'] = pd.to_timedelta(productsFlightsAnalysis['FlightTime']).dt.total_seconds()

# Convertir 'FlightTime' a minutos desde segundos
productsFlightsAnalysis['FlightTime'] = productsFlightsAnalysis['FlightTime'] / 60
productsFlightsAnalysis['Month'] = pd.to_datetime(productsFlightsAnalysis['STD']).dt.month
productsFlightsAnalysis['Day'] = pd.to_datetime(productsFlightsAnalysis['STD']).dt.day
productsFlightsAnalysis['Weekday'] = pd.to_datetime(productsFlightsAnalysis['STD']).dt.weekday
productsFlightsAnalysis['PartOfDay'] = le.fit_transform(productsFlightsAnalysis['PartOfDay'])
productsFlightsAnalysis = productsFlightsAnalysis.drop('STD', axis = 1)
productsFlightsAnalysis['DepartureStation' ] = le.fit_transform(productsFlightsAnalysis['DepartureStation'])
productsFlightsAnalysis['ArrivalStation'] = le.fit_transform(productsFlightsAnalysis['ArrivalStation'])

#%%
flights23.to_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Flights2023.csv')
productsPerFlight.to_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Productos por vuelo')
productsFlightsAnalysis.to_csv('Training Set No Normalizado.csv')

