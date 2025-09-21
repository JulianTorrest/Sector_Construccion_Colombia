import os
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, silhouette_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# Configuración de la página
st.set_page_config(page_title="GEIH 2024 - KPIs y ML", layout="wide")

# Ruta por defecto del CSV final
DEFAULT_CSV = Path(__file__).parent / "Unificados_final" / "final.csv"

st.title("GEIH 2024 - KPIs, Gráficos y Machine Learning")

########################################
# Sidebar - Selección de archivo y carga
########################################
st.sidebar.header("Datos y Filtros")

csv_path = st.sidebar.text_input("Ruta del archivo final.csv", value=str(DEFAULT_CSV))
max_rows = st.sidebar.number_input("Máx. filas a cargar (0 = todas)", min_value=0, value=0, step=100000)
use_pyarrow = st.sidebar.checkbox("Usar motor PyArrow (si está instalado)", value=True)
sample_pct = st.sidebar.slider("Muestrear porcentaje de filas", min_value=10, max_value=100, value=100, step=5)
sample_seed = st.sidebar.number_input("Semilla del muestreo", min_value=0, value=42, step=1)

def get_header(path: str):
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8') as f:
        line = f.readline().strip('\n\r')
    return [h.strip() for h in line.split(',')] if line else []

all_columns = get_header(csv_path)
limit_columns = False
usecols = None
if all_columns:
    limit_columns = st.sidebar.checkbox("Cargar solo columnas seleccionadas (optimiza memoria)", value=False)
    if limit_columns:
        usecols = st.sidebar.multiselect("Selecciona columnas a cargar", options=all_columns, default=all_columns[:50])

@st.cache_data(show_spinner=True)
def load_data(path: str, usecols=None, nrows: int | None = None, use_pyarrow: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.warning(f"No se encontró el archivo en: {p}")
        return pd.DataFrame()
    # Forzar dtype=str para conservar ceros a la izquierda en identificadores
    read_kwargs = dict(dtype=str, keep_default_na=False, low_memory=False)
    if usecols:
        read_kwargs["usecols"] = usecols
    if nrows and nrows > 0:
        read_kwargs["nrows"] = int(nrows)
    if use_pyarrow:
        try:
            read_kwargs["engine"] = "pyarrow"
            read_kwargs["dtype_backend"] = "pyarrow"
            df = pd.read_csv(p, **read_kwargs)
        except Exception as e:
            st.warning(f"No se pudo usar PyArrow ({e}). Se usará el motor por defecto de pandas.")
            read_kwargs.pop("engine", None)
            read_kwargs.pop("dtype_backend", None)
            df = pd.read_csv(p, **read_kwargs)
    else:
        df = pd.read_csv(p, **read_kwargs)
    # Intentar convertir algunas columnas numéricas conocidas sin perder strings vacíos
    for col in ["PERIODO", "MES", "PER", "DPTO", "AREA", "CLASE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    # Reducir memoria: convertir categóricas clave a category
    for c in ["DIRECTORIO", "SECUENCIA_P", "ORDEN", "DPTO", "AREA", "CLASE"]:
        if c in df.columns:
            try:
                df[c] = df[c].astype("category")
            except Exception:
                pass
    return df

df = load_data(csv_path, usecols=usecols, nrows=(None if max_rows == 0 else max_rows), use_pyarrow=use_pyarrow)

# Aplicar muestreo posterior a la carga si se indicó porcentaje < 100
if 10 <= sample_pct < 100 and len(df) > 0:
    frac = sample_pct / 100.0
    df = df.sample(frac=frac, random_state=sample_seed)

if df.empty:
    st.stop()

# Detectar columnas clave
keys = [c for c in ["DIRECTORIO", "SECUENCIA_P", "ORDEN"] if c in df.columns]

with st.sidebar:
    st.markdown("**Columnas detectadas:**")
    st.write(keys if keys else "No se detectaron las columnas clave.")

# Filtros dinámicos
cols_filters: List[str] = []
for col in ["PERIODO", "MES", "DPTO", "AREA", "CLASE"]:
    if col in df.columns:
        cols_filters.append(col)

filter_expander = st.sidebar.expander("Filtros (opcionales)", expanded=False)
with filter_expander:
    filtered_df = df.copy()
    for col in cols_filters:
        uniques = sorted([x for x in filtered_df[col].unique() if x != ""])
        if len(uniques) > 0:
            sel = st.multiselect(f"Filtrar {col}", options=uniques, default=[])
            if sel:
                filtered_df = filtered_df[filtered_df[col].isin(sel)]

st.markdown("## KPIs")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Registros (filas)", f"{len(filtered_df):,}" )
with kpi2:
    hogares = filtered_df.drop_duplicates(subset=[c for c in ["DIRECTORIO", "SECUENCIA_P"] if c in filtered_df.columns])
    st.metric("Hogares únicos", f"{len(hogares):,}")
with kpi3:
    personas = filtered_df.drop_duplicates(subset=[c for c in ["DIRECTORIO", "SECUENCIA_P", "ORDEN"] if c in filtered_df.columns])
    st.metric("Personas únicas", f"{len(personas):,}")
with kpi4:
    # KPI genérico: número de departamentos
    if "DPTO" in filtered_df.columns:
        st.metric("Departamentos", f"{filtered_df['DPTO'].nunique():,}")
    else:
        st.metric("Variables", f"{filtered_df.shape[1]:,}")

st.markdown("---")

st.markdown("## Gráficos")

g1, g2 = st.columns(2)

with g1:
    if "DPTO" in filtered_df.columns:
        # Usar operaciones sobre Series para evitar copias grandes de DataFrame
        s = filtered_df["DPTO"]
        s = s[s.astype(str) != ""]
        dpto_counts = s.value_counts(dropna=False).reset_index()
        dpto_counts.columns = ["DPTO", "conteo"]
        fig = px.bar(dpto_counts, x="DPTO", y="conteo", title="Registros por Departamento")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se encontró la columna DPTO para graficar.")

with g2:
    if "AREA" in filtered_df.columns:
        s = filtered_df["AREA"]
        s = s[s.astype(str) != ""]
        area_counts = s.value_counts(dropna=False).reset_index()
        area_counts.columns = ["AREA", "conteo"]
        fig2 = px.pie(area_counts, names="AREA", values="conteo", title="Distribución por AREA")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No se encontró la columna AREA para graficar.")

# Gráfico adicional: Ingresos por DPTO (si INGLABO existe)
st.markdown("### Ingresos por DPTO (agregación)")
if "DPTO" in filtered_df.columns:
    # Seleccionar métrica (por defecto INGLABO si existe; si no, elegir numérica)
    numeric_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(filtered_df[c], errors="coerce"))]
    default_metric = "INGLABO" if "INGLABO" in filtered_df.columns else (numeric_cols[0] if numeric_cols else None)
    metric = st.selectbox("Métrica numérica", options=numeric_cols, index=(numeric_cols.index(default_metric) if default_metric in numeric_cols else 0) if numeric_cols else None)
    agg = st.selectbox("Agregación", options=["mean", "median", "sum", "count"], index=0)
    topn = st.slider("Top N", 5, 50, 20)
    if metric:
        df_num = filtered_df.copy()
        df_num[metric] = pd.to_numeric(df_num[metric], errors="coerce")
        grouped = df_num.groupby("DPTO")[metric].agg(agg).reset_index(name=f"{agg}_{metric}")
        grouped = grouped.sort_values(by=f"{agg}_{metric}", ascending=False).head(topn)
        fig_inc = px.bar(grouped, x="DPTO", y=f"{agg}_{metric}", title=f"{agg.upper()} de {metric} por DPTO")
        st.plotly_chart(fig_inc, use_container_width=True)
else:
    st.info("No se puede calcular ingresos por DPTO sin la columna DPTO.")

# Constructor de tasas por grupo (por DPTO o AREA)
st.markdown("### Constructor de tasas")
group_dim = st.selectbox("Agrupar por", options=[opt for opt in ["DPTO", "AREA"] if opt in filtered_df.columns])
indicator_col = st.selectbox("Columna indicador (binaria o categórica)", options=sorted(filtered_df.columns))
success_value = st.text_input("Valor de éxito (se contará como 1)", value="1")
if group_dim and indicator_col:
    num = (filtered_df[indicator_col].astype(str) == str(success_value))
    rate_df = (
        pd.DataFrame({group_dim: filtered_df[group_dim].values, "success": num.astype(int)})
        .groupby(group_dim)["success"].mean()
        .reset_index(name="tasa")
        .sort_values("tasa", ascending=False)
    )
    fig_rate = px.bar(rate_df, x=group_dim, y="tasa", title=f"Tasa de {indicator_col} == {success_value} por {group_dim}")
    st.plotly_chart(fig_rate, use_container_width=True)

st.markdown("---")

st.markdown("## Explorador de variables")

with st.expander("Tabla (vista previa)", expanded=False):
    st.dataframe(filtered_df.head(200))
    # Exportar datos filtrados
    # Proteger exportación para evitar consumo de memoria excesivo
    approx_cells = int(filtered_df.shape[0]) * int(filtered_df.shape[1])
    if approx_cells <= 20_000_000:  # ~20M celdas
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar dataset filtrado (CSV)", data=csv_bytes, file_name="dataset_filtrado.csv", mime="text/csv")
    else:
        st.info("El dataset filtrado es muy grande para exportarlo completo. Te ofrecemos una muestra de 100.000 filas.")
        sample = filtered_df.sample(n=min(100_000, len(filtered_df)), random_state=42) if len(filtered_df) > 0 else filtered_df
        csv_bytes = sample.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar muestra (100k) CSV", data=csv_bytes, file_name="dataset_filtrado_sample.csv", mime="text/csv")

with st.expander("Frecuencias de variables", expanded=False):
    col_to_freq = st.selectbox("Selecciona una columna", options=sorted(filtered_df.columns))
    top_n = st.slider("Top N", 5, 50, 20)
    freq = (
        filtered_df[col_to_freq]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": col_to_freq, col_to_freq: "conteo"})
        .head(top_n)
    )
    st.write(freq)
    fig3 = px.bar(freq, x=col_to_freq, y="conteo", title=f"Top {top_n} valores de {col_to_freq}")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.markdown("## Machine Learning (demo rápida)")

ml_tab1, ml_tab2 = st.tabs(["Clasificación/Regresión", "Configuración y ayuda"]) 

with ml_tab1:
    # Selección de target
    target = st.selectbox("Selecciona variable objetivo (y)", options=[c for c in filtered_df.columns if c not in keys])

    # Heurística para tipo de problema
    if target:
        y_series = filtered_df[target]
        # Convertir a numérico si procede
        y_num = pd.to_numeric(y_series, errors="coerce")
        n_unique = y_series.nunique(dropna=False)
        is_classification = (y_series.dtype == object) or (n_unique <= 20 and y_num.notna().mean() < 0.9)

        # Selección de features
        default_features = [
            c for c in filtered_df.columns 
            if c not in keys + [target] and pd.api.types.is_numeric_dtype(pd.to_numeric(filtered_df[c], errors="coerce"))
        ]
        features = st.multiselect(
            "Selecciona variables predictoras (X)",
            options=[c for c in filtered_df.columns if c != target],
            default=default_features[:20],
        )
        # Opciones de ingeniería de características
        with st.expander("Opciones avanzadas (ingeniería y selección de variables)", expanded=False):
            categorical_candidates = [c for c in filtered_df.columns if filtered_df[c].dtype == object and c not in keys + [target]]
            one_hot_cols = st.multiselect("Columnas para one-hot encoding", options=categorical_candidates, default=[])
            kbest = st.number_input("Seleccionar top-K features (0 = desactivar)", min_value=0, max_value=200, value=0, step=1)

        if features:
            base_cols = keys + features + [target] if keys else features + [target]
            df_ml = filtered_df[base_cols].copy()

            # One-hot encoding si se solicitó
            if one_hot_cols:
                df_ml = pd.get_dummies(df_ml, columns=one_hot_cols, drop_first=True)
                # Actualizar features con las nuevas columnas dummies
                features = [c for c in df_ml.columns if c not in (keys + [target])]

            # Convertir X a numeric
            X = df_ml[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            # y: si clasificación -> string; si regresión -> numérico
            if is_classification:
                y = df_ml[target].astype(str)
                model = Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("clf", LogisticRegression(max_iter=1000))
                ])
            else:
                y = pd.to_numeric(df_ml[target], errors="coerce").fillna(0.0)
                model = Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("reg", LinearRegression())
                ])

            # Selección de características opcional
            if kbest and kbest > 0:
                try:
                    if is_classification:
                        selector = SelectKBest(score_func=mutual_info_classif, k=min(kbest, X.shape[1]))
                    else:
                        selector = SelectKBest(score_func=f_regression, k=min(kbest, X.shape[1]))
                    X = selector.fit_transform(X, y)
                except Exception as e:
                    st.warning(f"No fue posible aplicar SelectKBest: {e}")

            test_size = st.slider("Proporción de test", 0.1, 0.5, 0.2)
            random_state = st.number_input("random_state", min_value=0, value=42, step=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            if st.button("Entrenar modelo"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if is_classification:
                    acc = accuracy_score(y_test, y_pred)
                    try:
                        f1 = f1_score(y_test, y_pred, average="weighted")
                    except Exception:
                        f1 = float('nan')
                    st.success(f"Exactitud (accuracy): {acc:.3f} | F1 ponderado: {f1:.3f}")
                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.success(f"MAE: {mae:,.3f} | R²: {r2:.3f}")

                # Exportar predicciones
                try:
                    preds_df = pd.DataFrame({"y_true": y_test})
                    preds_df["y_pred"] = y_pred
                    pred_csv = preds_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar predicciones (CSV)", data=pred_csv, file_name="predicciones.csv", mime="text/csv")
                except Exception:
                    pass

with ml_tab2:
    st.markdown(
        """
        - Carga `Unificados_final/final.csv` automáticamente.
        - Filtra por variables básicas (PERIODO, MES, DPTO, AREA, CLASE) si existen.
        - KPIs: registros, hogares únicos, personas únicas.
        - Gráficos: barras por DPTO, torta por AREA, frecuencias de variables.
        - ML: Selecciona la variable objetivo y algunas predictoras numéricas; el sistema decide clasificación/regresión de forma heurística.
        - Este módulo es demostrativo; se recomienda refinar features, ingeniería de variables y validación.
        """
    )

st.markdown("---")

st.markdown("## Analítica avanzada")

tab_cluster, tab_assoc = st.tabs(["Clustering", "Asociación (Reglas)"])

with tab_cluster:
    st.subheader("K-Means / MiniBatch K-Means con PCA (visualización)")
    # Selección de variables numéricas para clustering
    numeric_candidates = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(filtered_df[c], errors="coerce"))]
    vars_cluster = st.multiselect("Variables numéricas para agrupar", options=numeric_candidates, default=numeric_candidates[:10])
    k = st.slider("Número de clusters (K)", min_value=2, max_value=15, value=5)
    algorithm = st.selectbox("Algoritmo", options=["KMeans", "MiniBatchKMeans"], index=0)
    do_scale = st.checkbox("Estandarizar variables", value=True)
    do_pca = st.checkbox("Reducir dimensionalidad a 2D con PCA para visualizar", value=True)
    run_cluster = st.button("Ejecutar clustering")

    if run_cluster and vars_cluster:
        Xc = filtered_df[vars_cluster].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scaler = StandardScaler(with_mean=False) if do_scale else None
        if scaler is not None:
            Xc_scaled = scaler.fit_transform(Xc)
        else:
            Xc_scaled = Xc.values
        if algorithm == "KMeans":
            model_c = KMeans(n_clusters=k, n_init=10, random_state=42)
        else:
            model_c = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
        labels = model_c.fit_predict(Xc_scaled)
        st.success("Clustering completado")
        # Silhouette (muestra para evitar OOM)
        try:
            idx = slice(None)
            if len(Xc_scaled) > 50000:
                idx = list(range(0, len(Xc_scaled), max(1, len(Xc_scaled)//50000)))
            sil = silhouette_score(Xc_scaled[idx], labels[idx])
            st.info(f"Silhouette score (aprox): {sil:.3f}")
        except Exception:
            pass
        # PCA para visualización
        if do_pca:
            try:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(Xc_scaled)
                plot_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "cluster": labels.astype(str)})
                figc = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", title="Clusters (PCA 2D)")
                st.plotly_chart(figc, use_container_width=True)
            except Exception as e:
                st.warning(f"No fue posible proyectar con PCA: {e}")
        # Exportar etiquetas
        try:
            export_df = pd.DataFrame({"cluster": labels})
            if keys:
                for kcol in keys:
                    if kcol in filtered_df.columns:
                        export_df[kcol] = filtered_df[kcol].values
            cluster_csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar etiquetas de cluster (CSV)", data=cluster_csv, file_name="clusters.csv", mime="text/csv")
        except Exception:
            pass

with tab_assoc:
    st.subheader("Reglas de Asociación (Apriori)")
    if not MLXTEND_AVAILABLE:
        st.warning("mlxtend no está instalado. Ejecuta: pip install mlxtend")
    else:
        # Selección de columnas categóricas
        cat_candidates = [c for c in filtered_df.columns if filtered_df[c].dtype == object or str(filtered_df[c].dtype).startswith("category")]
        cols_assoc = st.multiselect("Columnas categóricas", options=sorted(cat_candidates), default=cat_candidates[:5])
        top_per_col = st.number_input("Top categorías por columna (resto = OTROS)", min_value=3, max_value=50, value=10)
        min_support = st.slider("Soporte mínimo", min_value=0.001, max_value=0.2, value=0.01, step=0.001)
        min_conf = st.slider("Confianza mínima", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        max_rules = st.number_input("Máximo de reglas a mostrar", min_value=10, max_value=5000, value=200, step=10)
        run_assoc = st.button("Generar reglas")

        if run_assoc and cols_assoc:
            # Construcción de transacciones one-hot por columna=valor (limitando categorías)
            df_tx = pd.DataFrame(index=filtered_df.index)
            for col in cols_assoc:
                s = filtered_df[col].astype(str)
                top_vals = s.value_counts().head(int(top_per_col)).index
                s = s.where(s.isin(top_vals), other="OTROS")
                dummies = pd.get_dummies(s, prefix=col, dtype=bool)
                df_tx = pd.concat([df_tx, dummies], axis=1)
            # Apriori
            try:
                freq = apriori(df_tx, min_support=min_support, use_colnames=True)
                rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                # Ordenar por lift y recortar
                rules = rules.sort_values(by="lift", ascending=False).head(int(max_rules))
                st.dataframe(rules)
                # Exportar reglas
                rules_csv = rules.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar reglas (CSV)", data=rules_csv, file_name="reglas_asociacion.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"No fue posible generar reglas: {e}")

st.markdown("---")

st.markdown("## Modelos avanzados y Reducción de Dimensión")

tab_rf, tab_dimred = st.tabs(["RandomForest", "Reducción de Dimensión (PCA/UMAP)"])

with tab_rf:
    st.subheader("RandomForest (Clasificación / Regresión)")
    # Selección de target y features
    target_rf = st.selectbox("Variable objetivo (y)", options=[c for c in filtered_df.columns])
    numeric_candidates_rf = [c for c in filtered_df.columns if c != target_rf and pd.api.types.is_numeric_dtype(pd.to_numeric(filtered_df[c], errors="coerce"))]
    features_rf = st.multiselect("Variables predictoras (numéricas)", options=numeric_candidates_rf, default=numeric_candidates_rf[:20])
    n_estimators = st.slider("Árboles (n_estimators)", min_value=50, max_value=500, value=200, step=50)
    max_depth = st.slider("Profundidad máxima (0 = None)", min_value=0, max_value=30, value=0, step=1)
    test_size_rf = st.slider("Proporción de test", 0.1, 0.5, 0.2)
    rs_rf = st.number_input("random_state", min_value=0, value=42, step=1)
    run_rf = st.button("Entrenar RandomForest")

    if run_rf and target_rf and features_rf:
        df_rf = filtered_df[features_rf + [target_rf]].copy()
        X = df_rf[features_rf].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_series = df_rf[target_rf]
        # Heurística tipo de problema
        y_num = pd.to_numeric(y_series, errors="coerce")
        n_unique = y_series.nunique(dropna=False)
        is_classification_rf = (y_series.dtype == object) or (n_unique <= 20 and y_num.notna().mean() < 0.9)

        X_train, X_test, y_train, y_test = train_test_split(X, y_series if is_classification_rf else y_num.fillna(0.0), test_size=test_size_rf, random_state=rs_rf)

        if is_classification_rf:
            model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=rs_rf, n_jobs=-1)
        else:
            model_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=rs_rf, n_jobs=-1)

        model_rf.fit(X_train, y_train)
        y_pred = model_rf.predict(X_test)
        if is_classification_rf:
            acc = accuracy_score(y_test, y_pred)
            try:
                f1 = f1_score(y_test, y_pred, average="weighted")
            except Exception:
                f1 = float('nan')
            st.success(f"RandomForest (Clasificación) -> Accuracy: {acc:.3f} | F1: {f1:.3f}")
        else:
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success(f"RandomForest (Regresión) -> MAE: {mae:,.3f} | R²: {r2:.3f}")

        # Importancia de variables
        try:
            importances = pd.Series(model_rf.feature_importances_, index=features_rf).sort_values(ascending=False).head(30)
            fig_imp = px.bar(importances[::-1], orientation='h', title='Importancia de variables (Top 30)')
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            pass

        # Exportar predicciones
        try:
            preds_df = pd.DataFrame({"y_true": y_test})
            preds_df["y_pred"] = y_pred
            pred_csv = preds_df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar predicciones RF (CSV)", data=pred_csv, file_name="predicciones_rf.csv", mime="text/csv")
        except Exception:
            pass

with tab_dimred:
    st.subheader("PCA y UMAP (opcional)")
    numeric_candidates_dr = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(filtered_df[c], errors="coerce"))]
    vars_dr = st.multiselect("Variables numéricas", options=numeric_candidates_dr, default=numeric_candidates_dr[:10])
    do_scale_dr = st.checkbox("Estandarizar", value=True)
    n_comp = st.slider("Componentes PCA", min_value=2, max_value=10, value=3)
    run_pca = st.button("Ejecutar PCA")
    run_umap = st.button("Ejecutar UMAP (2D)")
    n_neighbors = st.slider("UMAP n_neighbors", min_value=5, max_value=100, value=15)
    min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1)

    if run_pca and vars_dr:
        Xd = filtered_df[vars_dr].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scaler = StandardScaler(with_mean=False) if do_scale_dr else None
        Xd_scaled = scaler.fit_transform(Xd) if scaler is not None else Xd.values
        try:
            pca = PCA(n_components=n_comp, random_state=42)
            pcs = pca.fit_transform(Xd_scaled)
            exp = pca.explained_variance_ratio_
            exp_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(exp))], "VarianzaExp": exp})
            fig_exp = px.bar(exp_df, x="PC", y="VarianzaExp", title="Varianza explicada por componente")
            st.plotly_chart(fig_exp, use_container_width=True)
            # Scatter 2D si n_comp>=2
            plot_df = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1]})
            st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", title="PCA 2D"), use_container_width=True)
            # Exportar PCs
            pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
            if keys:
                for kcol in keys:
                    if kcol in filtered_df.columns:
                        pcs_df[kcol] = filtered_df[kcol].values
            pcs_csv = pcs_df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar componentes PCA (CSV)", data=pcs_csv, file_name="pca_componentes.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"No fue posible ejecutar PCA: {e}")

    if run_umap and vars_dr:
        if not UMAP_AVAILABLE:
            st.warning("UMAP no está instalado. Ejecuta: pip install umap-learn")
        else:
            Xd = filtered_df[vars_dr].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            scaler = StandardScaler(with_mean=False) if do_scale_dr else None
            Xd_scaled = scaler.fit_transform(Xd) if scaler is not None else Xd.values
            try:
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
                emb = reducer.fit_transform(Xd_scaled)
                emb_df = pd.DataFrame({"U1": emb[:,0], "U2": emb[:,1]})
                st.plotly_chart(px.scatter(emb_df, x="U1", y="U2", title="UMAP 2D"), use_container_width=True)
                emb_csv = emb_df.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar embedding UMAP (CSV)", data=emb_csv, file_name="umap_embedding.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"No fue posible ejecutar UMAP: {e}")

st.markdown("---")

st.markdown("## Documentación")

tab_docs = st.tabs(["Documentación"])[0]
with tab_docs:
    readme_path = Path(__file__).parent / "README.md"
    negocio_path = Path(__file__).parent / "NEGOCIO.md"
    st.subheader("README (técnico)")
    try:
        st.markdown(readme_path.read_text(encoding='utf-8'))
    except Exception as e:
        st.warning(f"No se pudo leer README.md: {e}")
    st.subheader("Guía de Negocio")
    try:
        st.markdown(negocio_path.read_text(encoding='utf-8'))
    except Exception as e:
        st.warning(f"No se pudo leer NEGOCIO.md: {e}")
