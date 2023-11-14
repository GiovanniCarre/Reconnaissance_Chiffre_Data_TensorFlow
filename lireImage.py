import cv2
import numpy as np
import tensorflow as tf

# Charger l'image
image_path = "test3.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))  # Assurez-vous que l'image a la même résolution que les données d'entraînement
image = image.astype('float32') / 255  # Normaliser les valeurs des pixels

# Remodeler l'image pour correspondre à la forme d'entrée du modèle
image = np.reshape(image, (1, 28, 28, 1))

# Charger le modèle entraîné
model = tf.keras.models.load_model("mon_modele.h5")

# Faire la prédiction
predictions = model.predict(image)

# Afficher les résultats
predicted_class = np.argmax(predictions)
print(f"Le modèle prédit que le chiffre est : {predicted_class}")
