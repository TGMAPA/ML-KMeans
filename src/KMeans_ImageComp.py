# Miguel Ángel Pérez Ávila
# Archivo main con parte 2: Compresión bird_small

# Importación de librerías
import cv2 
import numpy as np
from KMeans import *

# Lectura de archivo con opencv
path = "../data/bird_small.png"
img_array = cv2.imread(path)

# Convertir matriz de BGR a RGB, ya que la lectura se hace BGR automaticaente
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# Aplanar la matriz para obtener un vector de (128x128)x3
pointsData = matrix2Flat(img_array)

# Definir k clusters deseados
k = 8

# Inicializar Centroides aleatoriamente
centroids = kMeansInitCentroids(pointsData, k)

# Ejecutar Kmeans y obtener el historial con la trayectoria de los centroides y las asignaciones
# de clusters para cada dato. Se ingresan los centroides inicializados anteriormente
dataCentroidsIdx, historyCentroidsCoords = runkMeans(pointsData, centroids, 60,  drawCentroids=True)

# Centroides obtenidos
color_centroids = np.array(historyCentroidsCoords[-1])
print("Centroides Obtenidos: ", color_centroids.shape)

compressedImage = []
# Remplazar puntos por centroide que generaliza su codificación RGB
for i in range(len(dataCentroidsIdx)):
    compressedImage.append(color_centroids[dataCentroidsIdx[i]])

# Nueva imagen
compressedImage = np.array(compressedImage)

# Redimensionar al tamaño original
compressedImage = np.reshape( compressedImage, shape=img_array.shape)

# Convertir a uint8 para rango válido 0–255
compressedImage = np.clip(compressedImage, 0, 255).astype(np.uint8)

# ----- Mostrar imagen analizada

# Crear un subplot con 1 fila y 2 columnas
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar primera imagen
axes[0].imshow(img_array)
axes[0].set_title("Imagen Original")
axes[0].axis("off") 

# Mostrar segunda imagen
axes[1].imshow(compressedImage)
axes[1].set_title("Imagen Compresa a K: "+ str(k) + " Colores")
axes[1].axis("off")

# Ajustar espacios y mostrar
plt.tight_layout()
plt.show()
