import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# URLs de los archivos
url_1 = 'https://raw.githubusercontent.com/JulianTorrest/Sector_Construccion_Colombia/main/Proyectos%20de%20Vivienda%20Nuevos/ws_fr_vn_tipos_ver2_cs.xlsx'
url_2 = 'https://raw.githubusercontent.com/JulianTorrest/Sector_Construccion_Colombia/main/Proyectos%20de%20Vivienda%20Nuevos/ws_fr_vn_ver2_cs.xlsx'

# --- 1. Carga y Limpieza de Datos ---
@st.cache_data
def load_and_clean_data(url_tipos, url_vn):
    """
    Carga, limpia y fusiona los dos DataFrames.
    """
    try:
        df_tipos = pd.read_excel(url_tipos)
        df_vn = pd.read_excel(url_vn)

        # Eliminar la columna 'Unnamed: 0' si existe
        if 'Unnamed: 0' in df_tipos.columns:
            df_tipos = df_tipos.drop(columns=['Unnamed: 0'])
        if 'Unnamed: 0' in df_vn.columns:
            df_vn = df_vn.drop(columns=['Unnamed: 0'])

        # Unir los DataFrames por la columna 'Link'
        df_merged = pd.merge(df_vn, df_tipos, on='Link', how='inner', suffixes=('_vn', '_tipos'))

        # Limpieza y conversión de tipos
        def clean_numeric_column(df, column_name):
            if column_name in df.columns:
                df[column_name] = df[column_name].astype(str).str.replace('$', '').str.replace('.', '', regex=False).str.replace(',', '').str.replace('m²', '').str.strip()
                # Reemplazar valores no numéricos con NaN
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            return df
        
        # Aplicar la limpieza a las columnas de precio y área
        df_merged = clean_numeric_column(df_merged, 'Precio_vn')
        df_merged = clean_numeric_column(df_merged, 'Precio_tipos')
        df_merged = clean_numeric_column(df_merged, 'Área')
        df_merged = clean_numeric_column(df_merged, 'Área construida')
        df_merged = clean_numeric_column(df_merged, 'Área privada')

        # Unificar las columnas de Precio y Área y eliminar las redundantes
        df_merged['Precio'] = df_merged['Precio_vn'].fillna(df_merged['Precio_tipos'])
        df_merged['Área'] = df_merged['Área'].fillna(df_merged['Área construida']).fillna(df_merged['Área privada'])
        df_merged = df_merged.drop(columns=['Precio_vn', 'Precio_tipos', 'Área construida', 'Área privada'], errors='ignore')

        # Rellenar valores nulos en columnas numéricas clave
        for col in ['Alcobas', 'Baños', 'Parqueaderos']:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0).astype('int64')

        # Eliminar filas con precios y áreas nulos
        df_merged.dropna(subset=['Precio', 'Área'], inplace=True)

        return df_merged

    except Exception as e:
        st.error(f"Error loading or cleaning data: {e}")
        return None

# Cargar los DataFrames limpios
df_final = load_and_clean_data(url_1, url_2)

# --- 2. Dashboard Interactivo ---
st.set_page_config(layout="wide")
st.title("🏡 Análisis de Proyectos de Vivienda en Colombia")

if df_final is not None and not df_final.empty:
    
    # Crea un DataFrame con la información de las columnas
    st.subheader('Campos y Tipos de Datos (Después de la Limpieza)')
    df_info_final = pd.DataFrame({
        'Campo': df_final.columns,
        'Tipo de Dato': df_final.dtypes
    })
    st.dataframe(df_info_final)
    
    st.subheader('Vista Previa de los Datos')
    st.dataframe(df_final.head())
    
    # Crear pestañas para organizar el contenido
    tab1, tab2, tab3 = st.tabs(["📊 KPIs y Resumen", "📈 Gráficos Interactivos", "🧠 Machine Learning"])

    with tab1:
        st.header("Indicadores Clave (KPIs)")
        
        # Columnas para los KPIs
        kpi1, kpi2, kpi3 = st.columns(3)

        total_proyectos = len(df_final)
        precio_promedio = df_final['Precio'].mean()
        precio_por_metro_cuadrado = (df_final['Precio'] / df_final['Área']).mean()

        kpi1.metric("Proyectos Analizados", f"{total_proyectos:,}")
        kpi2.metric("Precio Promedio", f"${precio_promedio:,.0f} COP")
        kpi3.metric("Precio Promedio por m²", f"${precio_por_metro_cuadrado:,.0f} COP")

        st.write("---")
        st.subheader("Distribución General")
        
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.write("**Proyectos por Ciudad**")
            proyectos_por_ciudad = df_final['Ciudad'].value_counts().head(10).reset_index()
            proyectos_por_ciudad.columns = ['Ciudad', 'Número de Proyectos']
            st.dataframe(proyectos_por_ciudad)
        
        with col_dist2:
            st.write("**Proyectos por Constructora**")
            proyectos_por_constructora = df_final['Constructora'].value_counts().head(10).reset_index()
            proyectos_por_constructora.columns = ['Constructora', 'Número de Proyectos']
            st.dataframe(proyectos_por_constructora)


    with tab2:
        st.header("Análisis Exploratorio y Gráficos Interactivos")
        
        # Filtros interactivos
        st.subheader("Filtros")
        ciudades = ['Todas'] + sorted(df_final['Ciudad'].dropna().unique().tolist())
        constructora_seleccionada = ['Todas'] + sorted(df_final['Constructora'].dropna().unique().tolist())
        
        col_filtro1, col_filtro2 = st.columns(2)
        ciudad_filtro = col_filtro1.selectbox("Selecciona una Ciudad", ciudades)
        constructora_filtro = col_filtro2.selectbox("Selecciona una Constructora", constructora_seleccionada)
        
        df_filtrado = df_final.copy()
        if ciudad_filtro != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Ciudad'] == ciudad_filtro]
        if constructora_filtro != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Constructora'] == constructora_filtro]
            
        if not df_filtrado.empty:
            
            # Gráfico 1: Scatter Plot de Precio vs. Área
            st.write("---")
            st.subheader("Gráfico de Dispersión: Precio vs. Área")
            
            c = alt.Chart(df_filtrado).mark_circle(size=60).encode(
                x=alt.X('Área', axis=alt.Axis(title='Área (m²)', format='~s')),
                y=alt.Y('Precio', axis=alt.Axis(title='Precio (COP)', format='~s')),
                color='Ciudad'
            ).properties(
                title='Precio vs. Área de la Vivienda'
            ).interactive()
            st.altair_chart(c, use_container_width=True)

            # Gráfico 2: Histograma de Precios
            st.write("---")
            st.subheader("Distribución de Precios")
            hist_precios = alt.Chart(df_filtrado).mark_bar().encode(
                x=alt.X("Precio", bin=alt.Bin(maxbins=50), title='Precio (COP)'),
                y=alt.Y('count()', title='Número de Proyectos')
            ).properties(
                title='Distribución de Precios de Viviendas'
            )
            st.altair_chart(hist_precios, use_container_width=True)
            
            # Gráfico 3: Proyectos por Estrato
            st.write("---")
            st.subheader("Proyectos por Estrato")
            df_estratos = df_filtrado['Estrato'].value_counts().reset_index()
            df_estratos.columns = ['Estrato', 'Número de Proyectos']
            bar_estratos = alt.Chart(df_estratos).mark_bar().encode(
                x=alt.X('Estrato:N', sort='-y', axis=alt.Axis(title='Estrato')),
                y=alt.Y('Número de Proyectos', title='Número de Proyectos')
            ).properties(
                title='Distribución de Proyectos por Estrato'
            )
            st.altair_chart(bar_estratos, use_container_width=True)

    with tab3:
        st.header("Hoja de Ruta para Machine Learning")
        st.write("Con los datos limpios y unificados, el siguiente paso es construir un **modelo de regresión** para predecir el precio de las viviendas.")
        st.image("https://i.imgur.com/L127bFz.png")

        st.subheader("Pasos a Seguir:")
        st.write("1. **Preparación de Datos:**")
        st.write("   - **Selección de Variables:** Identifica las variables que influyen en el precio: `Área`, `Alcobas`, `Baños`, `Parqueaderos`, `Estrato`, `Ciudad`, `Zona`, etc.")
        st.write("   - **Codificación:** Convierte las variables categóricas como `Ciudad` y `Zona` en un formato numérico usando técnicas como `One-Hot Encoding`.")
        st.write("2. **División de Datos:** Divide el conjunto de datos en un grupo de entrenamiento (para que el modelo aprenda) y un grupo de prueba (para evaluar su precisión).")
        st.write("3. **Entrenamiento del Modelo:**")
        st.write("   - Elige un algoritmo de regresión (por ejemplo, `Linear Regression`, `Random Forest Regressor` o `Gradient Boosting Regressor`).")
        st.write("   - Entrena el modelo con los datos de entrenamiento.")
        st.write("4. **Evaluación del Modelo:**")
        st.write("   - Usa métricas como `Mean Absolute Error (MAE)` o `R-squared` en los datos de prueba para ver qué tan bien predice el precio.")
        st.write("5. **Despliegue en Streamlit:**")
        st.write("   - Crea una interfaz en tu dashboard donde el usuario pueda ingresar las características de una vivienda y obtener una predicción de precio en tiempo real.")
