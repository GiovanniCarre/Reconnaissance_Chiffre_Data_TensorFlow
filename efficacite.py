import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Charger les données MNIST
(_, _), (test_images, test_labels) = mnist.load_data()

# Prétraitement des données de test
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_labels = tf.keras.utils.to_categorical(test_labels)

# Charger le modèle entraîné
model = tf.keras.models.load_model("mon_modele.h5")

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Précision sur l'ensemble de test : {accuracy * 100:.2f}%")
