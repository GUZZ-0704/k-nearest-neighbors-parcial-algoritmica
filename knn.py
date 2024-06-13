import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar el dataset
anime = pd.read_csv('anime.csv', sep='\t', keep_default_na=False)

print(anime.isnull().sum())
print(anime.head())

# Selección de características y objetivo
caracteristicas = ['num_episodes', 'score', 'score_count', 'favorites_count']
objetivo = 'members_count'

# Manejo de valores faltantes
anime_limpio = anime.dropna(subset=caracteristicas + [objetivo])
print(f'Datos después de eliminar filas con valores faltantes: {anime_limpio.shape}')

# Eliminar géneros repetidos dentro de cada fila
anime_limpio['genres'] = anime_limpio['genres'].apply(lambda x: '|'.join(sorted(set(x.split('|')))))

# Expandir las listas de géneros a columnas binarias
generos = anime_limpio['genres'].str.get_dummies(sep='|')
anime_limpio = anime_limpio.join(generos)

print('Datos con columnas de géneros añadidas:')
print(anime_limpio.head())

# Asegurar que no haya valores vacíos en las columnas seleccionadas
anime_limpio[caracteristicas] = anime_limpio[caracteristicas].apply(pd.to_numeric, errors='coerce')
anime_limpio.dropna(subset=caracteristicas, inplace=True)

caracteristicas = caracteristicas + generos.columns.tolist()
X = anime_limpio[caracteristicas]
y = anime_limpio[objetivo]

# Normalización de datos
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# División de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_escalado, y, test_size=0.2, random_state=42)

# Crear el modelo k-NN
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = knn_regressor.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse}')

# Popularidad por género individual
popularidad_genero_suma = anime_limpio[generos.columns.tolist() + [objetivo]].groupby(generos.columns.tolist()).sum()
popularidad_genero_suma = popularidad_genero_suma.sum().sort_values(ascending=False)

print('Popularidad por género (suma):')
print(popularidad_genero_suma.head())

# Lista para almacenar los valores de MSE para diferentes valores de k
mse_values = []

# Valores de k a probar
k_values = [1, 3, 5, 7, 9]

for k in k_values:
    # Crear el modelo k-NN
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = knn_regressor.predict(X_test)

    # Calcular el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Graficar los valores de MSE en función de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_values, marker='o')
plt.title('Error Cuadrático Medio vs. Número de Vecinos (k)')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.xticks(k_values)
plt.grid(True)
plt.show()

