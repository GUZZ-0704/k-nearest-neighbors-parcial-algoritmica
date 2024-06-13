import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Leer los datos
data = pd.read_csv('population_by_country_2020.csv', sep=',', keep_default_na=False)


data = data.replace('N.A.', np.nan)

data['Yearly Change'] = data['Yearly Change'].str.replace('%', '').astype(float) / 100
data['Urban Pop %'] = data['Urban Pop %'].str.replace('%', '').astype(float) / 100

data['Urban Pop %'] = data['Urban Pop %'].fillna(data['Urban Pop %'].median())


caracteristicas = ['Yearly Change', 'Density (P/Km²)', 'Land Area (Km²)',
                   'Fert. Rate', 'Med. Age', 'Urban Pop %']
objetivo = 'Population (2020)'

data_limpio = data.dropna(subset=caracteristicas + [objetivo])

print(data.isnull().sum())

print(data.shape)
print(data_limpio.shape)

print(data['Population (2020)'].head())

