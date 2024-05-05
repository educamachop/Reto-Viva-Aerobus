#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:46:34 2024

@author: samuelpelaez
"""

import pandas as pd
import numpy as np

#%%

NN = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Predicciones Productos NNS.csv')
NNS = pd.read_csv('/Users/samuelpelaez/Documents/Datathon Vivaerobus/Predicciones Productos NN.csv')


#%%

NN = NN.drop('Unnamed: 0', axis = 1)
NNS = NNS.drop('Unnamed: 0', axis = 1)
#%%
product_columns1 = NN.columns.difference(['Flight_ID', 'Aeronave', 'DepartureStation', 'ArrivalStation',
       'Destination_Type', 'Origin_Type', 'STD', 'STA', 'Capacity',
       'Passengers','FlightTime', 'Aforo', 'DepartureStation','ArrivalStation', 'PartOfDay', 'Month', 'Day', 'Weekday'])
product_columns2 = NNS.columns.difference(['Flight_ID', 'Aeronave', 'DepartureStation', 'ArrivalStation',
       'Destination_Type', 'Origin_Type', 'STD', 'STA', 'Capacity',
       'Passengers','FlightTime', 'Aforo', 'DepartureStation','ArrivalStation', 'PartOfDay', 'Month', 'Day', 'Weekday','Productos Totales'])





#%%
#RED NEURONAL SOLITA
# Lista para almacenar cada fila de productos vendidos
sold_products_list = []

# Iterar sobre cada fila del DataFrame original
for index, row in NN.iterrows():
    # Iterar sobre cada columna de producto en la fila
    for product in product_columns1:
        if row[product] > 0:  # Solo considerar productos con ventas
            sold_products_list.append({
                'Flight_ID': row['Flight_ID'],
                'Producto': product,
                'Cantidad': row[product],
                'Passengers': row['Passengers']
            })

# Convertir la lista en un DataFrame
sold_products_df_NN = pd.DataFrame(sold_products_list)

#%%

sold_products_df_NN['Passengers'] = np.floor(sold_products_df_NN['Passengers']).astype(int)

#%%

#LA DE MUCHAS REDES
# Lista para almacenar cada fila de productos vendidos
sold_products_list2 = []

# Iterar sobre cada fila del DataFrame original
for index, row in NNS.iterrows():
    # Iterar sobre cada columna de producto en la fila
    for product in product_columns2:
        if row[product] > 0:  # Solo considerar productos con ventas
            sold_products_list2.append({
                'Flight_ID': row['Flight_ID'],
                'Producto': product,
                'Cantidad': row[product],
                'Passengers': row['Passengers']
            })

# Convertir la lista en un DataFrame
sold_products_df_NNS = pd.DataFrame(sold_products_list2)

#%%

sold_products_df_NNS['Passengers'] = np.floor(sold_products_df_NN['Passengers']).astype(int)

#%%

sold_products_df_NNS.to_csv('Productos Varias Redes Neuronales.csv')
sold_products_df_NN.to_csv('Productos Una Red Neuronal.csv')



