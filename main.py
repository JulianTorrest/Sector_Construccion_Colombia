import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

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

        if 'Unnamed: 0' in df_tipos.columns:
            df_tipos = df_tipos.drop(columns=['Unnamed: 0'])
        if 'Unnamed: 0' in df_vn.columns:
            df_vn = df_vn.drop(columns=['Unnamed: 0'])

        df_merged = pd.merge(df_vn, df_tipos, on='Link', how='inner', suffixes=('_vn', '_tipos'))

        def clean_numeric_column(df, column_name):
            if column_name in df.columns:
                df[column_name] = df[column_name].astype(str).str.replace('$', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '').str.replace('m²', '').str.strip()
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            return df
        
        df_merged = clean_numeric_column(df_merged, 'Precio_vn')
        df_merged = clean_numeric_column(df_merged, 'Precio_tipos')
        df_merged = clean_numeric_column(df_merged, 'Área')
        df_merged = clean_numeric_column(df_merged, 'Área construida')
        df_merged = clean_numeric_column(df_merged, 'Área privada')

        df_merged['Precio'] = df_merged['Precio_vn'].fillna(df_merged['Precio_tipos'])
        df_merged['Área'] = df_merged['Área'].fillna(df_merged['Área construida']).fillna(df_merged['Área privada'])
        df_merged = df_merged.drop(columns=['Precio_vn', 'Precio_tipos', 'Área construida', 'Área privada'], errors='ignore')

        for col in ['Alcobas', 'Baños', 'Parqueaderos']:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0).astype('int64')

        df_merged.dropna(subset=['Precio', 'Área'], inplace=True)
        
        # Filtrar para evitar valores extremos que distorsionan el modelo
        df_merged = df_merged[(df_merged['Precio'] > 10000000) & (df_merged['Área'] > 10)]

        return df_merged

# --- 2. Entrenamiento del modelo de Machine Learning ---
@st.cache_resource
def train_model(df):
    """
    Entrena un modelo de regresión lineal y devuelve el modelo y las variables.
    """
    st.subheader("Entrenando el modelo de Machine Learning...")
    
    # 1. Selección de Variables
    features = ['Área', 'Alcobas', 'Baños', 'Parqueaderos', 'Estrato', 'Ciudad', 'Zona']
    target = 'Precio'
    
    # Asegurarse de que las columnas existen y no tienen nulos para el modelo
    df_model = df[features + [target]].dropna()
    
    X = df_model[features]
    y = df_model[target]

    # 2. Codificación para variables categóricas
    numeric_features = ['Área', 'Alcobas', 'Baños', 'Parqueaderos']
    categorical_features = ['Estrato', 'Ciudad', 'Zona']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 3. Creación del pipeline del modelo
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])
    
    # 4. División de los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Entrenamiento del modelo
    model_pipeline.fit(X_train, y_train)
    
    # 6. Evaluación del modelo
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    st.write(f"✅ **Modelo Entrenado Exitosamente**")
    st.write(f"R-squared (R²): {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): ${mae:,.0f} COP")
    st.write("---")
    
    return model_pipeline, features

# --- Main App ---
st.set_page_config(layout="wide")
st.title("🏡 Análisis y Predicción de Precios de Vivienda en Colombia")

# Cargar los DataFrames limpios
df_final = load_and_clean_data(url_1, url_2)

if df_final is not None and not df_final.empty:
    
    # Entrenar el modelo al inicio de la aplicación
    model, features = train_model(df_final)
    
    # --- Interfaz del Dashboard ---
    tab1, tab2, tab3 = st.tabs(["📊 KPIs y Resumen", "📈 Gráficos Interactivos", "🧠 Predicción de Precios"])

    with tab1:
        st.header("Indicadores Clave (KPIs)")
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
            st.write("---")
            st.subheader("Gráfico de Dispersión: Promedio de Precio vs. Promedio de Área por Ciudad")
            c_promedios = alt.Chart(df_filtrado).mark_circle(size=100).encode(
                x=alt.X('mean(Área)', axis=alt.Axis(title='Área Promedio (m²)', format='~s')),
                y=alt.Y('mean(Precio)', axis=alt.Axis(title='Precio Promedio (COP)', format='~s')),
                tooltip=[alt.Tooltip('Ciudad', title='Ciudad'), alt.Tooltip('mean(Área)', title='Área Promedio', format='~s'), alt.Tooltip('mean(Precio)', title='Precio Promedio', format='~s')]
            ).properties(
                title='Relación entre Precio y Área Promedio por Ciudad'
            ).interactive()
            st.altair_chart(c_promedios, use_container_width=True)
            
            st.write("---")
            st.subheader("Distribución de Precios")
            hist_precios = alt.Chart(df_filtrado).mark_bar().encode(
                x=alt.X("Precio", bin=alt.Bin(maxbins=50), title='Precio (COP)'),
                y=alt.Y('count()', title='Número de Proyectos')
            ).properties(
                title='Distribución de Precios de Viviendas'
            )
            st.altair_chart(hist_precios, use_container_width=True)
            
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
        st.header("Predicción del Precio de tu Vivienda")
        st.write("Ingresa las características de la vivienda y el modelo predecirá el precio.")
        
        # Obtener las listas de valores únicos para los selectboxes
        ciudades = sorted(df_final['Ciudad'].dropna().unique().tolist())
        zonas = sorted(df_final['Zona'].dropna().unique().tolist())
        estratos = sorted(df_final['Estrato'].dropna().unique().tolist())
        
        # Widgets para la entrada de datos del usuario
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Área (m²)", min_value=10, value=70, step=5)
            alcobas = st.selectbox("Alcobas", [1, 2, 3, 4, 5])
        with col2:
            banos = st.selectbox("Baños", [1, 2, 3, 4, 5])
            parqueaderos = st.selectbox("Parqueaderos", [0, 1, 2, 3, 4, 5])
        with col3:
            ciudad = st.selectbox("Ciudad", ciudades)
            zona = st.selectbox("Zona", zonas)
            estrato = st.selectbox("Estrato", estratos)
            
        st.write("---")
        
        if st.button("Predecir Precio"):
            try:
                # Crear un DataFrame con los datos de entrada del usuario
                data_input = pd.DataFrame([[area, alcobas, banos, parqueaderos, estrato, ciudad, zona]],
                                           columns=features)
                
                # Realizar la predicción
                prediction = model.predict(data_input)[0]
                
                st.success(f"📈 El precio estimado de la vivienda es: **${prediction:,.0f} COP**")
                
            except Exception as e:
                st.error(f"Ocurrió un error en la predicción: {e}")

else:
    st.warning("No se pudieron cargar o limpiar los datos. Por favor, revisa las URLs o el formato de los archivos.")
