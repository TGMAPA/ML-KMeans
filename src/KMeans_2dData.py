# Miguel Ángel Pérez Ávila
# Archivo main con parte 1: Clustering ex7data2

# Importación de librerías
import numpy as np
from KMeans import *

# Lectura de archivo
file = open("../data/ex7data2.txt", "r")

# Vector para almacenar datos
pointsData = []

# Almacenar datos leidos del archivo
for line in file.readlines():
    aux = line[:-1].split(" ")
    del aux[0]
    for i in range(len(aux)):
        aux[i] = float(aux[i])
    pointsData.append(aux)

# Cast de list a np.array omitiendo los ultimos dos valores del archivo de texto que parecen ser 
# saltos de línea
pointsData = np.array(pointsData[:-2])

# Definir K cluster deseados
k = 3

# Inicializar centroides de forma aleatoria
centroids = kMeansInitCentroids(pointsData, k)

# Ejecutar Kmeans y obtener el historial con la trayectoria de los centroides y las asignaciones
# de clusters para cada dato. Se ingresan los centroides inicializados anteriormente
dataCentroidsIdx, historyCentroidsCoords = runkMeans(pointsData, centroids, 20, drawCentroids=True)
