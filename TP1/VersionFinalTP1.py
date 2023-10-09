import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement de votre ensemble de données depuis un fichier CSV (par exemple)
# Assurez-vous d'avoir le bon chemin vers votre fichier CSV
data = np.loadtxt("Donnees_Numeriques.csv", delimiter=";", skiprows=1)

# Séparation des caractéristiques (features) et des étiquettes (labels)
X = data[:, :-1]  # Les colonnes 0 et 1 sont les caractéristiques
y = data[:, -1]   # La dernière colonne est l'étiquette (0 ou 1)

# Diviser les données en ensembles d'entraînement et de test (par exemple, 80% pour l'entraînement et 20% pour les tests)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = keras.Sequential([
    keras.layers.Dense(19, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(19, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle sur l'ensemble d'entraînement
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calcul de l'exactitude (accuracy) sur l'ensemble de test
accuracy = accuracy_score(y_test, y_pred_binary)

# Affichage de la matrice de confusion et du rapport de classification
confusion = confusion_matrix(y_test, y_pred_binary)
report = classification_report(y_test, y_pred_binary)

print("Matrice de confusion :\n", confusion)
print("Rapport de classification :\n", report)

# Affichage des résultats
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title('Données d\'entraînement')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.title('Données de test')

plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'], label='Précision')
plt.plot(history.history['val_accuracy'], label='Précision de validation')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.title('Évolution de la précision pendant l\'entraînement')
plt.show()
