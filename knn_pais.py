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


X = data_limpio[caracteristicas]
y = data_limpio[objetivo]


escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_escalado, y, train_size=0.8, test_size=0.2, random_state=42)


knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)


y_pred = knn_regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse}')



plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Línea base')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.title('Predicción de población utilizando k-NN')
plt.legend()
plt.show()

