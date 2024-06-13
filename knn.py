import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Cargar el dataset
anime = pd.read_csv('anime.csv', sep='\t', keep_default_na=False)

# Selección de características y objetivo
caracteristicas = ['score', 'score_count']
objetivo = 'popularity_rank'

# Manejo de valores faltantes
anime_limpio = anime.dropna(subset=caracteristicas + [objetivo])
print(f'Datos después de eliminar filas con valores faltantes: {anime_limpio.shape}')

# Asegurar que no haya valores vacíos en las columnas seleccionadas
anime_limpio[caracteristicas] = anime_limpio[caracteristicas].apply(pd.to_numeric, errors='coerce')
anime_limpio.dropna(subset=caracteristicas, inplace=True)

X = anime_limpio[caracteristicas]
y = anime_limpio[objetivo]

# Normalización de datos
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# División de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_escalado, y, test_size=0.2, random_state=42)

# Crear el modelo k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Crear una malla para visualizar las regiones de decisión
h = 0.02  # Tamaño del paso en la malla
x_min, x_max = X_escalado[:, 0].min() - 1, X_escalado[:, 0].max() + 1
y_min, y_max = X_escalado[:, 1].min() - 1, X_escalado[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar las regiones de decisión
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_escalado[:, 0], X_escalado[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('score')
plt.ylabel('score_count')
plt.title('K-NN en el Dataset de Anime')
plt.show()

# Predecir en el conjunto de prueba
y_pred = knn_classifier.predict(X_test)

# Mostrar algunas de las predicciones
print("Predicciones:")
print(y_pred[:10])  # Muestra las primeras 10 predicciones
print("Valores reales:")
print(y_test[:10])  # Muestra los primeros 10 valores reales


