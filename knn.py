import pandas as pd
import numpy as np

anime = pd.read_csv('anime.csv', sep='\t', keep_default_na=False)
print(anime.tail())

print(anime.isnull().sum())

# Selección de características y objetivo
model_anime = anime[['anime_id', 'title', 'synopsis', 'source_type', 'num_episodes', 'season', 'studios', 'genres', 'members_count']]
print(model_anime.head())
print(model_anime.shape)

# Selección de características y objetivo
caracteristicas = ['num_episodes', 'num_episodes', 'season', 'studios']
objetivo = 'members_count'

# Manejo de valores faltantes
anime_limpio = model_anime.dropna(subset=caracteristicas + [objetivo])
print(f'Datos después de eliminar filas con valores faltantes: {anime_limpio.shape}')

# División de los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = anime_limpio[caracteristicas]
y = anime_limpio[objetivo]







