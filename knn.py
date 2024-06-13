import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

pd.set_option('display.max_columns', None)

anime = pd.read_csv('anime.csv', sep='\t', keep_default_na=False)

anime_model = anime[['anime_id', 'title', 'synopsis', 'type', 'num_episodes', 'genres', 'score', 'members_count', 'popularity_rank']]

anime_model.loc[:, 'genres'] = anime_model['genres'].str.split('|')
anime_model = anime_model.explode('genres')
anime_model = pd.concat([anime_model, pd.get_dummies(anime_model['genres'])], axis=1)
anime_model.drop('genres', axis=1, inplace=True)

anime_model.dropna(inplace=True)

anime_model['num_episodes'] = pd.to_numeric(anime_model['num_episodes'], errors='coerce')
anime_model['score'] = pd.to_numeric(anime_model['score'], errors='coerce')
anime_model['popularity_rank'] = pd.to_numeric(anime_model['popularity_rank'], errors='coerce')
anime_model['members_count'] = pd.to_numeric(anime_model['members_count'], errors='coerce')

anime_model.dropna(inplace=True)

features = ['num_episodes', 'score', 'popularity_rank']
X = anime_model[features]
y = anime_model['members_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)

print("Predicciones y valores reales:")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())


accuracy = knn.score(X_test, y_test)
print("Precisión del modelo:", accuracy)



