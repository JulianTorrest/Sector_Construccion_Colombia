import streamlit as st
import pandas as pd

# URLs de los archivos
url_1 = 'https://raw.githubusercontent.com/JulianTorrest/Sector_Construccion_Colombia/main/Proyectos%20de%20Vivienda%20Nuevos/ws_fr_vn_tipos_ver2_cs.xlsx'
url_2 = 'https://raw.githubusercontent.com/JulianTorrest/Sector_Construccion_Colombia/main/Proyectos%20de%20Vivienda%20Nuevos/ws_fr_vn_ver2_cs.xlsx'

@st.cache_data
def load_data(url):
    try:
        df = pd.read_excel(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# Cargar los DataFrames
df_tipos = load_data(url_1)
df_vn = load_data(url_2)

# ---
# Generar y mostrar el listado de campos y tipos de datos
# ---

st.header("Análisis de Campos y Tipos de Datos")

if df_tipos is not None:
    st.subheader('ws_fr_vn_tipos_ver2_cs.xlsx')
    
    # Crea un DataFrame con la información de las columnas
    df_info_tipos = pd.DataFrame({
        'Campo': df_tipos.columns,
        'Tipo de Dato': df_tipos.dtypes
    })
    
    st.dataframe(df_info_tipos)

if df_vn is not None:
    st.subheader('ws_fr_vn_ver2_cs.xlsx')
    
    # Crea un DataFrame con la información de las columnas
    df_info_vn = pd.DataFrame({
        'Campo': df_vn.columns,
        'Tipo de Dato': df_vn.dtypes
    })
    
    st.dataframe(df_info_vn)
