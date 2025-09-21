# 🏡 Análisis y Predicción de Precios de Vivienda en Colombia

Este proyecto es una aplicación interactiva construida con **Streamlit** que utiliza técnicas de Machine Learning para analizar y predecir los precios de vivienda nueva en Colombia. La herramienta proporciona un dashboard interactivo con indicadores clave (KPIs), visualizaciones de datos y un modelo predictivo que permite a los usuarios estimar el precio de una vivienda a partir de sus características.

## Características Principales

* **KPIs y Resumen**: Una visión general de alto nivel con métricas clave como el precio promedio, el área mediana y el número total de proyectos.
* **Gráficos Interactivos**: Explora la distribución de precios, la relación entre variables y el precio promedio por ciudad y zona.
* **Predicción de Precios**: Un modelo de regresión (Random Forest) que predice el precio de una vivienda basado en el área, número de alcobas, baños, parqueaderos, estrato, ciudad y zona.
* **Análisis Avanzado**: Visualiza agrupamientos (clusters) de proyectos y un análisis de componentes principales (PCA) para comprender la estructura subyacente de los datos.

## Tecnologías y Librerías

El proyecto está desarrollado en Python y utiliza las siguientes librerías principales:

* **Streamlit**: Para la creación de la interfaz de usuario interactiva y el dashboard.
* **Pandas**: Para la manipulación y limpieza de los datos.
* **NumPy**: Para operaciones numéricas eficientes.
* **Altair**: Para la creación de visualizaciones de datos declarativas y atractivas.
* **Scikit-learn**: Para el entrenamiento de los modelos de Machine Learning (Random Forest, K-Means, PCA).

## Datos del Proyecto

Los datos utilizados para este análisis provienen de dos archivos en formato Excel alojados en GitHub:

* **`ws_fr_vn_tipos_ver2_cs.xlsx`**: Contiene información detallada sobre los tipos de vivienda (áreas, alcobas, baños, etc.).
* **`ws_fr_vn_ver2_cs.xlsx`**: Incluye datos generales de los proyectos de vivienda (ciudad, zona, constructora, etc.).

El script `main.py` se encarga de cargar, limpiar y fusionar estos dos archivos para crear un conjunto de datos coherente para el análisis.

## Cómo Ejecutar la Aplicación

Para ejecutar esta aplicación en tu entorno local, sigue los siguientes pasos:

### 1. Requisitos Previos

Asegúrate de tener Python instalado en tu sistema. Se recomienda usar un entorno virtual.

### 2. Clonar el Repositorio

Clona el repositorio de GitHub donde se encuentra el código de tu proyecto.

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
