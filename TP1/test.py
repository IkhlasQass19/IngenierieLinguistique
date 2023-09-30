import pandas as pd
import numpy as np

def get_IO_data(file_path):
    # Lecture du fichier CSV en utilisant pandas
    df = pd.read_csv(file_path)

    # Stockage des données du CSV dans un tableau (array)
    data_array = df.to_numpy()

    # Mélange des lignes de la matrice
    np.random.shuffle(data_array)

    # Extraction de la colonne "_À" et stockage dans une liste
    _A_column = data_array[:, df.columns.get_loc("_À")]
    _A_list = _A_column.tolist()

    return data_array, _A_list

# Chemin vers votre fichier CSV
file_path = 'votre_fichier.csv'

# Appel de la fonction pour obtenir les données
data_array, _A_list = get_IO_data(file_path)
print(data_array)
# Maintenant, data_array contient toutes les données du CSV et _A_list contient la colonne "_À".
