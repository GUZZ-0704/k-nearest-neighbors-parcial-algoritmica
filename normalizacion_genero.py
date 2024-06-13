import pandas as pd
import numpy as np

# Leer los datos desde el archivo CSV
file_path = 'anime.csv'  # Reemplazar con la ruta correcta del archivo CSV
data = pd.read_csv(file_path,sep='\t', keep_default_na=False)

# Separar los géneros y aplanar la lista
all_genres = data['genres'].str.split('|').explode().unique()

# Crear un DataFrame de géneros únicos con ID
genre_df = pd.DataFrame(all_genres, columns=['genre']).reset_index()
genre_df.columns = ['id', 'genre']

# Mostrar el DataFrame resultante
print(genre_df)