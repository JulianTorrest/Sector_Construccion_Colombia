# GEIH 2024 – Guía para Usuarios de Negocio

Este documento describe, en lenguaje no técnico, qué puedes hacer con el dashboard de GEIH 2024 y cómo interpretar sus resultados principales. El objetivo es facilitar el uso para exploración, seguimiento y toma de decisiones.

## ¿Qué es este proyecto?

- Integra múltiples archivos de la encuesta GEIH en un único dataset consolidado.
- Ofrece un dashboard interactivo para consultar indicadores clave (KPIs), visualizar información y realizar análisis avanzados (segmentación, reglas de asociación y modelos predictivos sencillos) sin necesidad de programar.

## ¿Qué puedo ver en el dashboard?

- **Indicadores (KPIs)**
  - Registros totales (filas).
  - Hogares únicos.
  - Personas únicas.
  - Número de departamentos (si está disponible).

- **Gráficas**
  - Registros por departamento (DPTO).
  - Distribución por tipo de área (AREA).
  - Ingresos por departamento: puedes seleccionar la métrica (por ejemplo, el ingreso laboral) y el tipo de agregación (promedio, mediana, suma, conteo).
  - Constructor de tasas: crea un indicador binario (por ejemplo, “empleado sí/no”) y compara la tasa por DPTO o AREA.

- **Exploración de variables**
  - Vista previa de la tabla.
  - Frecuencias de valores para cualquier variable.
  - Descarga del dataset filtrado (si es muy grande, se ofrece una muestra).

- **Analítica avanzada**
  - **Clustering (segmentación)**: agrupa registros similares de acuerdo con variables seleccionadas. Útil para identificar perfiles o segmentos.
  - **Reglas de asociación**: identifica combinaciones de categorías que ocurren juntas (por ejemplo, perfiles y características que suelen presentarse al mismo tiempo).

- **Modelos y reducción de datos**
  - **RandomForest**: modelos simples de clasificación (categorías) y regresión (valores numéricos) para evaluar qué variables explican mejor un resultado.
  - **PCA (Análisis de Componentes Principales)**: resume la información en menos variables; permite visualizar datos complejos en 2D.
  - **UMAP (opcional)**: técnica de proyección para explorar patrones no lineales.

## ¿Cómo usarlo en el día a día?

1. **Cargar datos**: el dashboard ya apunta al archivo consolidado `Unificados_final/final.csv`. Si necesitas otro archivo, puedes cambiar la ruta en el panel lateral.
2. **Filtrar**: limita el análisis por periodo, mes, departamento, área, etc.
3. **KPIs y gráficas**: revisa tendencias y comparaciones básicas (por ejemplo, ingresos por DPTO).
4. **Segmentación (Clustering)**: elige varias variables numéricas relevantes, ajusta el número de grupos y observa los patrones.
5. **Asociación**: selecciona variables categóricas y explora reglas con altos niveles de confianza y relevancia.
6. **Modelos**: prueba RandomForest para entender la importancia de variables y predecir resultados (demo), y descarga las predicciones si lo necesitas.

## Recomendaciones prácticas

- Empieza con pocas variables y una muestra de datos para obtener resultados rápidos. Aumenta gradualmente el volumen y la complejidad.
- En segmentación, prueba diferentes cantidades de grupos y revisa la métrica “Silhouette” para validar estabilidad de los clusters.
- En reglas de asociación, aumenta o reduce los umbrales de soporte/confianza para controlar la cantidad de reglas.
- Usa los botones de descarga para guardar resultados (reglas, etiquetas de cluster, predicciones) y compartirlos con tu equipo.

## Limitaciones y notas

- Este dashboard es una herramienta de exploración. Los modelos incluidos (RandomForest, logística, etc.) son demostrativos; para producción se recomienda un proceso riguroso de preparación de datos, validación y revisión de sesgos.
- Los resultados dependen de la calidad del dato de origen. Verifica definiciones y supuestos de las variables.
- Si los datos son muy grandes, el rendimiento puede verse afectado. Hay opciones de muestreo y selección de columnas para trabajar de manera fluida.

Si el archivo es muy grande, define “Máx. filas a cargar” (p. ej. 300000) y/o “Muestrear porcentaje de filas” (p. ej. 50%).
Si necesitas aún menos memoria, marca “Cargar solo columnas seleccionadas” y elige solo las columnas que usarás para los KPIs/gráficos/ML.

Sugerencias de configuración según tu RAM

8 GB: 150k–300k filas, 30–60% muestreo, columnas seleccionadas.
16 GB: 300k–800k filas, 50–100%, columnas seleccionadas si es necesario.
32+ GB: puedes intentar todo el dataset, pero aún así PyArrow y categorías ayudan.

Sugerencias de uso dentro del dashboard

En “Analítica avanzada” > “Clustering”:
Selecciona 5–15 variables numéricas.
Activa “Estandarizar variables”.
Elige K (empieza en 5), ejecuta y revisa el Silhouette. Ajusta K.
Activa PCA para ver clusters en 2D. Descarga las etiquetas si te sirven.
En “Analítica avanzada” > “Asociación (Reglas)”:
Selecciona 3–8 columnas categóricas o discretizadas.
Top categorías por columna controla dimensionalidad (10 suele ir bien).
Ajusta soporte/confianza según rendimiento. Empieza con soporte 0.01 y confianza 0.5.
Descarga reglas para analizarlas luego.
Consejos de rendimiento:
Si el CSV es muy grande, combina: “Máx. filas a cargar”, “Muestrear %”, y “Cargar solo columnas seleccionadas”.
Para reglas de asociación, menos columnas y categorías más frecuentes rinden mejor.
Para clustering, usa MiniBatchKMeans si ves lentitud y limita variables.

