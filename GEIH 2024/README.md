# GEIH 2024 – Unificación de datos y Dashboard de Análisis

Este proyecto unifica archivos CSV de la encuesta GEIH que están distribuidos en múltiples subcarpetas, los integra en un único dataset final y provee un dashboard interactivo (Streamlit) para KPIs, visualizaciones y modelos de Machine Learning.

## Estructura del proyecto

```
GEIH 2024/
├─ geih.py                     # Script de unificación y merge final
├─ app.py                      # Dashboard Streamlit (KPIs, gráficas y ML)
├─ requirements.txt            # Dependencias del proyecto
├─ Unificados/                 # Salida: CSV unificados por nombre (intermedios)
└─ Unificados_final/
   └─ final.csv               # Salida: dataset integrado (merge de todos los CSV)
```

## Requisitos

- Python 3.8+
- Windows PowerShell (para los comandos sugeridos)
- Paquetes de `requirements.txt` (pandas, streamlit, plotly, scikit-learn, pyarrow, mlxtend, umap-learn, etc.)

Se recomienda usar un entorno virtual `.venv` para aislar dependencias.

## Instalación y entorno

1) Activar el entorno virtual (opcional pero recomendado)

```powershell
& "C:\Users\betol\Downloads\GEIH 2024\.venv\Scripts\Activate.ps1"
python --version
```

2) Instalar dependencias

```powershell
pip install -r ".\requirements.txt"
```

Si deseas ejecutar el dashboard sin activar el entorno, usa el intérprete del venv directamente:

```powershell
& "C:\Users\betol\Downloads\GEIH 2024\.venv\Scripts\python.exe" -m pip install -r "C:\Users\betol\Downloads\GEIH 2024\requirements.txt"
```

## Unificación de archivos (geih.py)

`geih.py` recorre recursivamente la raíz, agrupa por nombre “similar” (normalización que une variantes como `No ocupado`/`No ocupados`), detecta delimitadores/codificaciones y concatena en `Unificados/` escribiendo el encabezado una vez. Luego puede realizar un merge final en `Unificados_final/final.csv` usando llaves detectadas automáticamente.

Comando típico:

```powershell
python geih.py --root "." --out-dir "Unificados" --final-merge --final-out "Unificados_final/final.csv"
```

Notas importantes:
- Si `--final-merge` está activo y ya existe `Unificados_final/final.csv`, se omite la unificación final.
- Llaves de unión detectadas: `DIRECTORIO`, `SECUENCIA_P` y `ORDEN` (persona); si un archivo no tiene `ORDEN`, se une por hogar (DIRECTORIO+SECUENCIA_P) o por `DIRECTORIO`.
- Delimitador detectado por archivo (`,`, `;`, `\t`, `|`). Salida normalizada a coma.
- Codificaciones probadas: `utf-8`, `latin-1`, `cp1252` (con reintentos al fallar en lectura). Puedes extender con `--encodings`.

## Dashboard (app.py)

El dashboard carga `Unificados_final/final.csv` por defecto y ofrece:

- KPIs: filas totales, hogares únicos, personas únicas.
- Gráficas: barras por `DPTO`, pie por `AREA`, ingresos por `DPTO` (agregación: mean/median/sum/count), constructor de tasas por `DPTO`/`AREA`.
- Explorador: vista previa de tabla, frecuencias de variables, descarga del dataset filtrado (o muestra si es muy grande).
- ML (demo): clasificación/regresión (LogisticRegression/LinearRegression), ingeniería (one‑hot), SelectKBest, métricas y exportación de predicciones.
- Analítica avanzada:
  - Clustering: KMeans/MiniBatchKMeans con PCA 2D y Silhouette (muestra), descarga de etiquetas.
  - Reglas de asociación: `mlxtend` (apriori/association_rules) con recorte de categorías y descarga de reglas.
- Modelos avanzados y Reducción de dimensión:
  - RandomForest (clasificación/regresión) con importancia de variables y descarga de predicciones.
  - PCA: varianza explicada, scatter 2D y descarga de componentes.
  - UMAP (opcional): proyección 2D y descarga del embedding.

### Ejecutar el dashboard

Con el entorno virtual activo:

```powershell
python -m streamlit run app.py
```

Alternativa sin activar el entorno (llamando al intérprete del venv):

```powershell
& "C:\Users\betol\Downloads\GEIH 2024\.venv\Scripts\python.exe" -m streamlit run "C:\Users\betol\Downloads\GEIH 2024\app.py"
```

La aplicación abre en `http://localhost:8501`. Para detenerla: `Ctrl + C` en la consola.

## Sugerencias de rendimiento y memoria

- **Máx. filas a cargar**: configura un tope (ej. 300000) para no cargar todo en RAM.
- **Muestrear porcentaje de filas**: usa 30–60% para explorar rápido; sube si tu RAM lo permite.
- **Cargar solo columnas seleccionadas**: activa esta opción y elige únicamente las columnas necesarias para KPIs/gráficos/ML.
- **PyArrow**: deja activa la casilla “Usar motor PyArrow” (si está instalado) para lectura de CSV más eficiente.
- **Tipos categóricos**: el dashboard convierte columnas clave (`DIRECTORIO`, `SECUENCIA_P`, `ORDEN`, `DPTO`, `AREA`, `CLASE`) a `category` para reducir memoria.
- **Exportación**: si el dataset filtrado es muy grande, la app ofrece una **muestra de 100.000 filas** para descargar.

### Guía según RAM disponible

- 8 GB: 150k–300k filas, 30–60% muestreo, columnas seleccionadas.
- 16 GB: 300k–800k filas, 50–100%, columnas seleccionadas si es necesario.
- 32+ GB: puedes intentar todo el dataset; PyArrow y categorías mejoran el rendimiento.

## Sugerencias de uso dentro del dashboard

- **Clustering (Analítica avanzada > Clustering)**
  - Selecciona 5–15 variables numéricas.
  - Activa “Estandarizar variables”.
  - Elige K (empieza en 5), ejecuta y revisa el Silhouette. Ajusta K.
  - Activa PCA para ver clusters en 2D. Descarga etiquetas.

- **Asociación (Analítica avanzada > Asociación)**
  - Selecciona 3–8 columnas categóricas o discretizadas.
  - Top categorías por columna controla dimensionalidad (10 suele ir bien).
  - Ajusta soporte/confianza según rendimiento. Empieza con soporte 0.01 y confianza 0.5.
  - Descarga reglas para analizarlas luego.

- **Consejos de rendimiento**
  - Si el CSV es muy grande, combina: “Máx. filas a cargar”, “Muestrear %”, y “Cargar solo columnas seleccionadas”.
  - Para reglas de asociación, menos columnas y categorías más frecuentes rinden mejor.
  - Para clustering, usa MiniBatchKMeans si ves lentitud y limita variables.

## Solución de problemas (Troubleshooting)

- **streamlit no se reconoce como comando**:
  - Activa el venv o usa `python -m streamlit run app.py`.
  - Verifica instalación: `pip install streamlit`.

- **UnicodeDecodeError durante unificación**:
  - `geih.py` ya reintenta con múltiples codificaciones. Puedes añadir `--encodings "utf-8;utf-8-sig;cp1252;latin-1"`.

- **Errores de memoria (MemoryError / ArrayMemoryError)**:
  - Reduce “Máx. filas a cargar”.
  - Habilita “Muestrear porcentaje de filas”.
  - Activa “Cargar solo columnas seleccionadas”.
  - Mantén PyArrow activado.

## Comandos útiles (resumen)

```powershell
# Activar venv
& "C:\Users\betol\Downloads\GEIH 2024\.venv\Scripts\Activate.ps1"

# Instalar dependencias
pip install -r ".\requirements.txt"

# Unificar e integrar
python geih.py --root "." --out-dir "Unificados" --final-merge --final-out "Unificados_final/final.csv"

# Ejecutar dashboard
python -m streamlit run app.py
```

## Licencia

Uso interno/educativo. Ajusta la licencia según tus necesidades.
