import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

# Cargar el dataset
anime = pd.read_csv('anime-dataset-2023.csv', sep=',', keep_default_na=False)

# Seleccionar las columnas relevantes
anime_model = anime[['anime_id', 'Name', 'Genres', 'Type', 'Score', 'Episodes', 'Rank', 'Popularity', 'Members']]

# Dividir la columna 'Genres' en múltiples géneros
anime_model.loc[:, 'Genres'] = anime_model['Genres'].str.split(',')
anime_model = anime_model.explode('Genres')
anime_model = pd.concat([anime_model, pd.get_dummies(anime_model['Genres'])], axis=1)
anime_model.drop('Genres', axis=1, inplace=True)

# Reemplazar valores no numéricos en 'Episodes' con NaN y convertir a numérico
anime_model['Episodes'] = anime_model['Episodes'].replace('UNKNOWN', np.nan)
anime_model['Episodes'] = pd.to_numeric(anime_model['Episodes'], errors='coerce')

# Convertir otras columnas a numérico y manejar NaNs
anime_model['Score'] = pd.to_numeric(anime_model['Score'], errors='coerce')
anime_model['Rank'] = pd.to_numeric(anime_model['Rank'], errors='coerce')
anime_model['Popularity'] = pd.to_numeric(anime_model['Popularity'], errors='coerce')
anime_model['Members'] = pd.to_numeric(anime_model['Members'], errors='coerce')

# Eliminar filas con valores NaN
anime_model.dropna(inplace=True)

# Seleccionar las características y la variable objetivo
features = ['Score', 'Episodes', 'Popularity']
X = anime_model[features]
y = anime_model['Members']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo KNeighborsRegressor con búsqueda de hiperparámetros
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)

# Calcular el error absoluto medio (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Error absoluto medio:", mae)

# Mostrar las predicciones y los valores reales
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Predicciones y valores reales:")
print(results.head())

# Calcular la precisión del modelo
accuracy = best_knn.score(X_test, y_test)
print("Precisión del modelo:", accuracy)
