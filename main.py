import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

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

        df_merged['Estrato'] = pd.to_numeric(df_merged['Estrato'], errors='coerce')
        df_merged = df_merged[df_merged['Estrato'].isin([1, 2, 3, 4, 5, 6])].copy()
        df_merged['Estrato'] = df_merged['Estrato'].astype('int64')
        
        df_merged.dropna(subset=['Precio', 'Área', 'Ciudad', 'Zona'], inplace=True)
        
        df_merged = df_merged[(df_merged['Precio'] > 10000000) & (df_merged['Área'] > 10)]

        return df_merged

    except Exception as e:
        st.error(f"Error loading or cleaning data: {e}")
        return None

# --- Funciones para los Modelos de Machine Learning ---
@st.cache_resource
def train_regression_model(df):
    """
    Entrena un modelo de regresión de Bosque Aleatorio (Random Forest).
    """
    st.subheader("Entrenando el modelo de Machine Learning...")
    features = ['Área', 'Alcobas', 'Baños', 'Parqueaderos', 'Estrato', 'Ciudad', 'Zona']
    target = 'Precio'
    df_model = df[features + [target]].dropna()
    X = df_model[features]
    y = df_model[target]
    numeric_features = ['Área', 'Alcobas', 'Baños', 'Parqueaderos']
    categorical_features = ['Estrato', 'Ciudad', 'Zona']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"✅ **Modelo Entrenado Exitosamente**")
    st.write(f"R-squared (R²): {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): ${mae:,.0f} COP")
    st.write("---")
    return model_pipeline, features, X_test, y_test, y_pred

@st.cache_resource
def run_clustering(df):
    """
    Ejecuta el modelo de clustering K-Means.
    """
    st.subheader("Agrupamiento (Clustering) con K-Means")
    st.write("Se están agrupando los proyectos de vivienda en **4 clusters** basados en sus características.")
    features_for_clustering = ['Precio', 'Área', 'Alcobas', 'Baños', 'Parqueaderos']
    clustering_df = df[features_for_clustering + ['Ciudad', 'Zona']].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df[features_for_clustering])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    clustering_df['Cluster'] = clusters
    
    return clustering_df, features_for_clustering

@st.cache_resource
def run_pca_analysis(df):
    """
    Ejecuta el Análisis de Componentes Principales (PCA).
    """
    st.subheader("Reducción de Dimensión con PCA")
    st.write("Reduciendo la dimensionalidad del dataset para una mejor visualización de los datos.")
    features_for_pca = ['Precio', 'Área', 'Alcobas', 'Baños', 'Parqueaderos']
    pca_df = df[features_for_pca].dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pca_df)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    pca_results_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    
    st.write(f"Varianza explicada por los 2 componentes: {pca.explained_variance_ratio_.sum():.2f}")
    
    return pca_results_df

# --- Main App ---
st.set_page_config(layout="wide")
st.title("🏡 Análisis y Predicción de Precios de Vivienda en Colombia")

# Cargar los DataFrames limpios
df_final = load_and_clean_data(url_1, url_2)

if df_final is not None and not df_final.empty:
    
    model, features, X_test, y_test, y_pred = train_regression_model(df_final)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 KPIs y Resumen", "📈 Gráficos Interactivos", "🧠 Predicción de Precios", "🔬 Análisis Avanzado"])

    with tab1:
        st.header("Indicadores Clave (KPIs)")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi4, kpi5, kpi6 = st.columns(3)

        total_proyectos = len(df_final)
        precio_promedio = df_final['Precio'].mean()
        precio_por_metro_cuadrado = (df_final['Precio'] / df_final['Área']).mean()
        proyectos_sin_parqueadero = (df_final['Parqueaderos'] == 0).sum()
        mediana_area = df_final['Área'].median()
        proyectos_destacados = (df_final['Destacado'] == 'Sí').sum()

        kpi1.metric("Proyectos Analizados", f"{total_proyectos:,}")
        kpi2.metric("Precio Promedio", f"${precio_promedio:,.0f} COP")
        kpi3.metric("Precio Promedio por m²", f"${precio_por_metro_cuadrado:,.0f} COP")
        kpi4.metric("Mediana del Área", f"{mediana_area:,.0f} m²")
        kpi5.metric("Proyectos Destacados", f"{proyectos_destacados:,}")
        kpi6.metric("Proyectos sin Parqueadero", f"{proyectos_sin_parqueadero:,}")
        
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
            
            st.write("---")
            st.subheader("Precio Promedio por Estrato")
            df_precio_estrato = df_filtrado.groupby('Estrato')['Precio'].mean().reset_index()
            df_precio_estrato.columns = ['Estrato', 'Precio Promedio']
            bar_precio_estrato = alt.Chart(df_precio_estrato).mark_bar().encode(
                x=alt.X('Estrato:N', sort='-y', axis=alt.Axis(title='Estrato')),
                y=alt.Y('Precio Promedio', title='Precio Promedio (COP)', axis=alt.Axis(format='~s'))
            ).properties(
                title='Precio Promedio por Estrato'
            ).interactive()
            st.altair_chart(bar_precio_estrato, use_container_width=True)
            
            st.write("---")
            st.subheader("Precio Promedio por Número de Alcobas")
            df_precio_alcobas = df_filtrado.groupby('Alcobas')['Precio'].mean().reset_index()
            df_precio_alcobas.columns = ['Alcobas', 'Precio Promedio']
            bar_precio_alcobas = alt.Chart(df_precio_alcobas).mark_bar().encode(
                x=alt.X('Alcobas:N', sort=None, axis=alt.Axis(title='Número de Alcobas')),
                y=alt.Y('Precio Promedio', title='Precio Promedio (COP)', axis=alt.Axis(format='~s'))
            ).properties(
                title='Precio Promedio por Número de Alcobas'
            ).interactive()
            st.altair_chart(bar_precio_alcobas, use_container_width=True)

    with tab3:
        st.header("Predicción del Precio de tu Vivienda")
        st.write("Ingresa las características de la vivienda y el modelo predecirá el precio.")
        
        ciudades = sorted(df_final['Ciudad'].dropna().unique().tolist())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Área (m²)", min_value=10, value=70, step=5)
            alcobas = st.selectbox("Alcobas", [1, 2, 3, 4, 5])
        with col2:
            banos = st.selectbox("Baños", [1, 2, 3, 4, 5])
            parqueaderos = st.selectbox("Parqueaderos", [0, 1, 2, 3, 4, 5])
        with col3:
            ciudad_prediccion = st.selectbox("Ciudad", ciudades, key='ciudad_pred')
            
            if ciudad_prediccion:
                zonas_filtradas = sorted(df_final[df_final['Ciudad'] == ciudad_prediccion]['Zona'].dropna().unique().tolist())
                zona_prediccion = st.selectbox("Zona", zonas_filtradas, key='zona_pred')
            else:
                zona_prediccion = st.selectbox("Zona", [], key='zona_pred')
            
            estratos_validos = sorted(df_final['Estrato'].dropna().unique().tolist())
            estrato_prediccion = st.selectbox("Estrato", estratos_validos, key='estrato_pred')
            
        st.write("---")
        
        if st.button("Predecir Precio"):
            try:
                data_input = pd.DataFrame([[area, alcobas, banos, parqueaderos, estrato_prediccion, ciudad_prediccion, zona_prediccion]],
                                           columns=features)
                
                prediction = model.predict(data_input)[0]
                
                st.success(f"📈 El precio estimado de la vivienda es: **${prediction:,.0f} COP**")
                
            except Exception as e:
                st.error(f"Ocurrió un error en la predicción: {e}")
        
        st.write("---")
        st.subheader("Análisis de Predicciones del Modelo")
        
        city_mask = X_test['Ciudad'] == ciudad_prediccion
        y_test_filtered = y_test[city_mask]
        y_pred_filtered = y_pred[city_mask]
        test_data_filtered = X_test[city_mask]
        
        chart_data = pd.DataFrame({
            'Zona': test_data_filtered['Zona'],
            'Precio Real': y_test_filtered,
            'Precio Pronosticado': y_pred_filtered
        }).melt(id_vars='Zona', var_name='Tipo de Precio', value_name='Precio')

        chart = alt.Chart(chart_data).mark_circle(size=60).encode(
            x=alt.X('Zona', title='Zona', axis=alt.Axis(labels=False)),
            y=alt.Y('Precio', title='Precio (COP)', axis=alt.Axis(format='~s')),
            color=alt.Color('Tipo de Precio', title='Tipo de Precio'),
            tooltip=['Zona', alt.Tooltip('Precio', format='$,.0f')]
        ).properties(
            title=f'Comparación de Precios en {ciudad_prediccion}'
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    with tab4:
        st.header("Análisis Avanzado de Machine Learning")
        st.write("Esta sección presenta otros modelos de aprendizaje automático para una comprensión más profunda de los datos.")

        # --- Agrupamiento (Clustering) ---
        clustering_df, features_for_clustering = run_clustering(df_final)
        st.write("---")
        st.write("### Gráficos de Clusters (Agrupamientos)")
        
        ciudades_clustering = ['Todas'] + sorted(clustering_df['Ciudad'].dropna().unique().tolist())
        ciudad_cluster_filtro = st.selectbox("Filtrar por Ciudad", ciudades_clustering, key='ciudad_cluster_filtro')

        if ciudad_cluster_filtro != 'Todas':
            df_cluster_filtrado = clustering_df[clustering_df['Ciudad'] == ciudad_cluster_filtro]
        else:
            df_cluster_filtrado = clustering_df

        # --- Gráfico 1: Área vs. Precio ---
        st.write("#### 📈 Área vs. Precio")
        st.write("Visualiza la relación entre el área y el precio, con puntos coloreados por cluster.")
        cluster_chart_area_precio = alt.Chart(df_cluster_filtrado).mark_circle().encode(
            x=alt.X('Área', title='Área (m²)', axis=alt.Axis(format='~s')),
            y=alt.Y('Precio', title='Precio (COP)', axis=alt.Axis(format='~s')),
            color='Cluster:N',
            tooltip=['Cluster', 'Área', alt.Tooltip('Precio', format='$,.0f')]
        ).properties(
            title=f'Proyectos Agrupados por Área y Precio en {ciudad_cluster_filtro}'
        ).interactive()
        st.altair_chart(cluster_chart_area_precio, use_container_width=True)

        # --- Gráfico 2: Precio vs. Alcobas ---
        st.write("---")
        st.write("#### 🛌 Precio vs. Alcobas")
        st.write("Examina la relación entre el precio y el número de alcobas. Podrías encontrar clústeres de proyectos de alto valor con muchas alcobas.")
        cluster_chart_precio_alcobas = alt.Chart(df_cluster_filtrado).mark_circle().encode(
            x=alt.X('Alcobas', title='Número de Alcobas', axis=alt.Axis(format='d')),
            y=alt.Y('Precio', title='Precio (COP)', axis=alt.Axis(format='~s')),
            color='Cluster:N',
            tooltip=['Cluster', 'Alcobas', alt.Tooltip('Precio', format='$,.0f')]
        ).properties(
            title=f'Proyectos Agrupados por Precio y Alcobas en {ciudad_cluster_filtro}'
        ).interactive()
        st.altair_chart(cluster_chart_precio_alcobas, use_container_width=True)

        # --- Gráfico 3: Área vs. Baños ---
        st.write("---")
        st.write("#### 🛀 Área vs. Baños")
        st.write("Explora si los proyectos de mayor área tienden a tener más baños, una posible señal de lujo o confort.")
        cluster_chart_area_banos = alt.Chart(df_cluster_filtrado).mark_circle().encode(
            x=alt.X('Baños', title='Número de Baños', axis=alt.Axis(format='d')),
            y=alt.Y('Área', title='Área (m²)', axis=alt.Axis(format='~s')),
            color='Cluster:N',
            tooltip=['Cluster', 'Baños', alt.Tooltip('Área', format='~s')]
        ).properties(
            title=f'Proyectos Agrupados por Área y Baños en {ciudad_cluster_filtro}'
        ).interactive()
        st.altair_chart(cluster_chart_area_banos, use_container_width=True)

        # --- Gráfico 4: Alcobas vs. Parqueaderos ---
        st.write("---")
        st.write("#### 🅿️ Alcobas vs. Parqueaderos")
        st.write("Analiza si hay una correlación entre el número de alcobas y la disponibilidad de parqueaderos.")
        cluster_chart_alcobas_parqueaderos = alt.Chart(df_cluster_filtrado).mark_circle().encode(
            x=alt.X('Alcobas', title='Número de Alcobas', axis=alt.Axis(format='d')),
            y=alt.Y('Parqueaderos', title='Número de Parqueaderos', axis=alt.Axis(format='d')),
            color='Cluster:N',
            tooltip=['Cluster', 'Alcobas', 'Parqueaderos']
        ).properties(
            title=f'Proyectos Agrupados por Alcobas y Parqueaderos en {ciudad_cluster_filtro}'
        ).interactive()
        st.altair_chart(cluster_chart_alcobas_parqueaderos, use_container_width=True)

        # --- Gráfico de Reducción de Dimensión (PCA) ---
        st.write("---")
        pca_results_df = run_pca_analysis(df_final)
        st.write("### Gráfico de Reducción de Dimensión (PCA)")
        st.write("Este gráfico muestra los datos proyectados en 2 componentes principales. La cercanía entre los puntos indica similitud entre los proyectos.")
        
        pca_chart = alt.Chart(pca_results_df).mark_circle().encode(
            x=alt.X('Principal Component 1', title='Componente Principal 1'),
            y=alt.Y('Principal Component 2', title='Componente Principal 2'),
            tooltip=['Principal Component 1', 'Principal Component 2']
        ).properties(
            title='Análisis de Componentes Principales (PCA)'
        ).interactive()
        
        st.altair_chart(pca_chart, use_container_width=True)

else:
    st.warning("No se pudieron cargar o limpiar los datos. Por favor, revisa las URLs o el formato de los archivos.")
