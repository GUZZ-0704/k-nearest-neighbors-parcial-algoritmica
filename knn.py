import pandas as pd
import numpy as np

anime = pd.read_csv('anime.csv', sep = '\t', keep_default_na=False)
print(anime.tail())

print(anime.isnull().sum())

# Selección de características y objetivo
caracteristicas = ['num_episodes', 'score', 'score_count', 'favorites_count']
objetivo = 'members_count'

# Manejo de valores faltantes
anime_limpio = anime.dropna(subset=caracteristicas + [objetivo])
print(f'Datos después de eliminar filas con valores faltantes: {anime_limpio.shape}')

model_anime = anime[['anime_id', 'title', 'synopsis', 'source_type', 'num_episodes', 'season', 'studios', 'genres', 'members_count']]
print(model_anime.head())
print(model_anime.shape)




