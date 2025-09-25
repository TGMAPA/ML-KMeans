# ML-KMeans

Este proyecto implementa el algoritmo **K-Means** de manera manual en Python, sin depender de librerías de machine learning para el entrenamiento. Se incluye su aplicación tanto en **clustering de datos** como en **compresión de imágenes por reducción de colores**, ilustrando el proceso paso a paso.

---

## Descripción

El algoritmo K-Means busca particionar un conjunto de datos en **K clusters**. Cada punto se asigna al cluster con el centroide más cercano, actualizando iterativamente las posiciones de los centroides hasta alcanzar la convergencia.  
Este proyecto muestra cómo aplicar K-Means desde cero para:

- Agrupamiento de datos en dos dimensiones.  
- Compresión de imágenes reduciendo la cantidad de colores.  

---

## Metodología

1. **Inicialización**: selección aleatoria de *K* centroides.  
2. **Asignación**: cada punto se asigna al centroide más cercano (distancia euclidiana).  
3. **Actualización**: los centroides se recalculan como el promedio de los puntos asignados.  
4. **Iteración**: se repiten los pasos 2 y 3 hasta la convergencia.  
5. **Aplicación a imágenes**:  
   - Los píxeles se consideran puntos en el espacio RGB.  
   - Se reducen los colores de la imagen asignando cada píxel al color de su centroide.  

---

## Resultados

- **Clustering de datos**: visualización de clusters y centroides en 2D.  
- **Compresión de imágenes**: reducción de colores con K = 8, 16, etc., mostrando el balance entre calidad visual y tamaño de la imagen.  

---

## Tecnologías Utilizadas

Se emplearon las siguientes librerías y versiones de Python:
- **Python 3.11.13** 
- **NumPy** → operaciones matemáticas y manejo de arreglos.  
- **Matplotlib** → visualización de datos y clusters.  
- **cv2** → carga y procesamiento de imágenes.  

---
 ## Ejecución
1. Clonar este repositorio:  
   ```bash
   git clone https://github.com/TGMAPA/ML-LogisticRegression-Regularized.git
   cd ML-LogisticRegression-Regularized/src
   python main.py
