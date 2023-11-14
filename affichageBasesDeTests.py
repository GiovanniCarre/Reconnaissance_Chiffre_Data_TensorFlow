import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Charger les données MNIST
(_, _), (test_images, test_labels) = mnist.load_data()

# Afficher 50 exemples d'images
plt.figure(figsize=(15, 10))
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"Etiquette : {test_labels[i]}")
    plt.axis('off')

plt.show()
