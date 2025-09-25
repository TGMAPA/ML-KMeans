# Miguel Ángel Pérez Ávila 
# Archivo con funciones implementadas: KMeans

# Importación de librerías
import numpy as np
import matplotlib.pyplot as plt


# =========== Funciones ===========

# Función para inicializar aleatoriamente K Centroides indicados.
#
# - Input:
#       - X : Vector de datos
#       - k : Número de clusters deseado
# - Return : 
#       - centroids : list() -> lista de k centroides inicializados
def kMeansInitCentroids(X, k):
    index_array = np.arange(0, len(X))

    # Seleccionar k indices aleatoriamente sin repetirse
    centroids_idx = np.random.choice(index_array, size=k, replace=False)

    centroids = []
    for i in centroids_idx:
        centroids.append(X[i])

    return centroids

# Función para encontrar los centroides mas cercanos para cada punto.
#
# - Input:
#       - X : Vector de datos
#       - inital_centroids : Vector de centroides y su ubicación
# - Return : 
#       - dataCentroidsIdx : list() -> lista de clusters asociadas a cada iesima muestra
def findClosestCentroids(X, inital_centroids):
    dataCentroidsIdx = []

    # Recorrer todos los puntos
    for i in range(len(X)):
        min_dx_c_idx = 0
        min_dx_c = 0
        firstIteration = True

        # Calculr distancia entre el punto y los centroides
        for j in range(len(inital_centroids)):
            dx = calcDistance(X[i], inital_centroids[j])

            if firstIteration:
                firstIteration = False
                min_dx_c = dx
                min_dx_c_idx = j

            # Encontrar el centroide más cercano
            if dx < min_dx_c:
                min_dx_c = dx
                min_dx_c_idx = j
        
        # Asignar el indice del centroide más cercano al punto actual
        dataCentroidsIdx.append(min_dx_c_idx)
    
    # Devolver vector con índices de centroides asignados a cada punto
    return dataCentroidsIdx

# Función para modificar la ubicación de los centroides utilizando la media de los puntos.
#
# - Input:
#       - X : Vector de datos
#       - idx : vector de relación entre los datos y el cluster al que se encuentra asignado.
#       - K : número de k clusters
# - Return : 
#       - newCentroids : list() -> Vector actualizado de centroides y su ubicación
def computeCentroids(X, idx, K):
    newCentroids = []

    for k in range(K):
        acum = np.zeros(shape=(len(X[0]))) # Inicializar el vector de sumatorias del tamaño apropiado de dimensiones
        m_counter = 0
        for point, point_centroid in zip(X, idx):
            
            if k == point_centroid:
                m_counter+=1
                # Evaluar los puntos asignados al centroide k
            
                # Sumar en todas las dimensiones del punto
                acum+= point 
                
        # Aplicar división entre m muestras para calcular el promedio en cada dimensióń
        # Si el cluster no esta vació 
        if m_counter>0:
            acum = acum/m_counter# Sumar en todas las dimensiones del punto

        # Agregar la nueva ubicación del centroide
        newCentroids.append(acum)
        
    return newCentroids

# Función para ejecutar algoitmo kmeans.
#
# - Input:
#       - X : Vector de datos
#       - initial_centroids : Vector inicial de centroides y su ubicación
#       - max_iters : número iteraciones a ejecutar la actualización de centroides
#       - drawCentroids : bool para gráficar los puntos y la trayectoria de los centroides
# - Return : 
#       - historyCentroidsCoords : list() -> Vector que contiene la trayectoria de la ubicación de los centroides
#       - dataCentroidsIdx : list() -> vector de relación entre los datos y el cluster al que se encuentra asignado.
def runkMeans(X, initial_centroids, max_iters, drawCentroids = True):
    # Convertir datos a flotantes 64 
    X = X.astype(np.float64)

    # Inicializar los centroides con los centroides dados
    centroids = initial_centroids

    # Declarar K 
    K = len(centroids)

    # Vector con el historial de la trayectoria de los centroides
    historyCentroidsCoords = [initial_centroids]

    # Clusters relacionados con los puntos
    dataCentroidsIdx = []

    for _ in range(max_iters):
        # Asignar puntos al cluster más cercano
        dataCentroidsIdx = findClosestCentroids(X, centroids)

        # Calcula los nuevos centroides de acuerdo con la media de los puntos asignados
        centroids = computeCentroids(X, dataCentroidsIdx, K)

        # ALmacenar nuevos centroides
        historyCentroidsCoords.append(centroids)

    # Si se desea graficar los puntos y la trayectoria de los centroides
    if drawCentroids:
        plotKMeansWithTrajectories3d(X, dataCentroidsIdx, historyCentroidsCoords)

    return dataCentroidsIdx, historyCentroidsCoords

# =========== Fin Funciones ===========


# ----------- Matemáticas

# Funcion para aplanar una matriz
def matrix2Flat(matrix):
    vector = []

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            vector.append(matrix[i][j])
    
    return np.array(vector)

# Calcular la distancia entre dos vectores
def calcDistance(vector1, vector2):
    acum = 0
    for i,j in zip(vector1, vector2):
        acum+= (i-j)**2
    return np.sqrt(acum)

# ----------- Visualización

# Plotear con trayectoria de los centroides (2D y 3D)
def plotKMeansWithTrajectories3d(X, idx, historyCentroidsCoords):
    K = len(historyCentroidsCoords[0])
    cmap = plt.get_cmap("tab10", K)  # genera K colores distintos

    fig = plt.figure(figsize=(10, 8))

    if X.shape[1] == 2:
        # Gráfica 2d
        ax = fig.add_subplot(111)
        final_idx = np.array(idx)

        # Graficar puntos de acuerdo con el cluster final
        for k in range(K):
            cluster_points = X[final_idx == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       s=30, color=cmap(k), label=f"Cluster {k+1}", alpha=0.6)

        # Graficar trayectoria de centroides
        for k in range(K):
            trajectory = np.array([centroids[k] for centroids in historyCentroidsCoords])
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                    marker='X', markersize=10, linestyle='--',
                    color=cmap(k), linewidth=2, label=f"Trayectoria C{k+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    elif X.shape[1] == 3:
        # Gr´fica 3d
        ax = fig.add_subplot(111, projection='3d')
        final_idx = np.array(idx)

        # Graficar puntos de acuerdo con el cluster final
        for k in range(K):
            cluster_points = X[final_idx == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                       s=30, color=cmap(k), label=f"Cluster {k+1}", alpha=0.6)

        # Graficar trayectoria de centroides
        for k in range(K):
            trajectory = np.array([centroids[k] for centroids in historyCentroidsCoords])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    marker='X', markersize=10, linestyle='--',
                    color=cmap(k), linewidth=2, label=f"Trayectoria C{k+1}")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    else:
        print("Solo se soportan datos en 2D o 3D para visualización")
        exit(0)

    plt.title("K-Means Clustering con Trayectoria de Centroides")
    plt.legend()
    plt.show()