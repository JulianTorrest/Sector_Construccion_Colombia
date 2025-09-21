import streamlit as st
import pandas as pd

# The raw URLs are needed to read the files directly.
# The `blob` part of the URL needs to be replaced with `raw`.
url_1 = 'https://raw.githubusercontent.com/JulianTorrest/Sector_Construccion_Colombia/main/Proyectos%20de%20Vivienda%20Nuevos/ws_fr_vn_tipos_ver2_cs.xlsx'
url_2 = 'https://raw.githubusercontent.com/JulianTorrest/Sector_Construccion_Colombia/main/Proyectos%20de%20Vivienda%20Nuevos/ws_fr_vn_ver2_cs.xlsx'

# Function to read the Excel file and cache it to avoid re-downloading on every interaction.
@st.cache_data
def load_data(url):
    """
    Loads an Excel file from a URL into a pandas DataFrame.
    """
    try:
        df = pd.read_excel(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# Load the data from both URLs.
df_tipos = load_data(url_1)
df_vn = load_data(url_2)

# Display the dataframes if they were loaded successfully.
if df_tipos is not None:
    st.subheader('DataFrame de ws_fr_vn_tipos_ver2_cs.xlsx')
    st.dataframe(df_tipos.head()) # Display the first 5 rows
    st.write(f'Filas: {len(df_tipos)}, Columnas: {len(df_tipos.columns)}')

if df_vn is not None:
    st.subheader('DataFrame de ws_fr_vn_ver2_cs.xlsx')
    st.dataframe(df_vn.head()) # Display the first 5 rows
    st.write(f'Filas: {len(df_vn)}, Columnas: {len(df_vn.columns)}')

# Add a simple message to confirm the process.
st.success('Archivos Excel cargados exitosamente. 👍')
