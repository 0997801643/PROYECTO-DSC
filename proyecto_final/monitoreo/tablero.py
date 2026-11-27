"""
Tablero de monitoreo (sección 6.2 del PSet 4).

Lee de Snowflake la tabla PRODUCCION.MONITOREO (la llena el flujo de Kestra "monitoreo") y
muestra, lote por lote: alertas, reentrenamientos, negocio, desempeño, modelo, datos y sistema.
Los reentrenamientos se leen del servidor de MLflow (corridas con fase = reentrenamiento).

Se abre en http://localhost:8501 (servicio "tablero" del docker-compose).
"""
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Monitoreo · Merma alta", layout="wide")


@st.cache_data(ttl=60)
def cargar_monitoreo():
    import snowflake.connector
    from cryptography.hazmat.primitives import serialization
    llave = serialization.load_pem_private_key(
        Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"]).read_bytes(), password=None)
    with snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        role=os.environ["SNOWFLAKE_ROLE"], warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        private_key=llave.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()),
    ) as con:
        cur = con.cursor()
        cur.execute("""SELECT LOTE_SIMULACION, TIPO, METRICA, VARIABLE, VALOR, UMBRAL, ALERTA, DETALLE,
                              FECHA_EJECUCION
                       FROM PRODUCCION.MONITOREO ORDER BY LOTE_SIMULACION""")
        df = pd.DataFrame(cur.fetchall(), columns=[c[0].lower() for c in cur.description])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["lote"] = df["lote_simulacion"].astype(int)
    return df


@st.cache_data(ttl=60)
def cargar_reentrenamientos():
    """Corridas de reentrenamiento desde la API REST de MLflow (sin instalar mlflow en la imagen)."""
    base = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")
    experimento = os.environ.get("MLFLOW_EXPERIMENT", "merma-cerveza")

    def pedir(ruta, cuerpo=None):
        datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
        req = urllib.request.Request(base + ruta, data=datos, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    exp_id = pedir("/api/2.0/mlflow/experiments/get-by-name?experiment_name="
                   + urllib.parse.quote(experimento))["experiment"]["experiment_id"]
    corridas = pedir("/api/2.0/mlflow/runs/search", {"experiment_ids": [exp_id],
                     "filter": "tags.fase = 'reentrenamiento'", "max_results": 100}).get("runs", [])
    versiones = pedir("/api/2.0/mlflow/model-versions/search?filter="
                      + urllib.parse.quote("name='merma-cerveza-modelo'")).get("model_versions", [])
    por_run = {v["run_id"]: v for v in versiones}

    filas = []
    for c in corridas:
        tags = {t["key"]: t["value"] for t in c["data"].get("tags", [])}
        met = {m["key"]: m["value"] for m in c["data"].get("metrics", [])}
        version = por_run.get(c["info"]["run_id"], {})
        tags_v = {t["key"]: t["value"] for t in version.get("tags", [])}
        motivo = tags.get("motivo_disparo", "")
        lote = re.search(r"detectado en el lote (\d+)", motivo)
        if "rechazado_por" in tags_v:
            decision = f"rechazado por {tags_v['rechazado_por']}"
        elif "aprobado_por" in tags_v:
            decision = f"promovido por {tags_v['aprobado_por']}"
        else:
            decision = "sin decisión registrada"
        filas.append({
            "fecha": pd.to_datetime(c["info"]["start_time"], unit="ms"),
            "lote que lo disparó": int(lote.group(1)) if lote else None,
            "versión challenger": version.get("version"),
            "motivo": motivo,
            "recomendación": tags.get("recomendacion", ""),
            "decisión": decision,
            "costo/lote champion (USD)": met.get("champion_costo_por_lote"),
            "costo/lote challenger (USD)": met.get("challenger_costo_por_lote"),
        })
    return pd.DataFrame(filas).sort_values("fecha", ascending=False) if filas else pd.DataFrame()


def serie(df, tipo, metrica, variable=""):
    """Valores de una métrica por lote."""
    f = df[(df["tipo"] == tipo) & (df["metrica"] == metrica) & (df["variable"].fillna("") == variable)]
    return f.set_index("lote")["valor"].sort_index()


df = cargar_monitoreo()
if df.empty:
    st.warning("Todavía no hay métricas: ejecuta el flujo 'monitoreo' en Kestra.")
    st.stop()

ultimo = df["lote"].max()
alertas = df[df["alerta"]]

st.title("Monitoreo del modelo de merma alta")
st.caption(f"Lotes de simulación 1 a {ultimo} · una semana de cocción por lote · "
           f"última ejecución del monitoreo: {df['fecha_ejecucion'].max()}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alertas en el último lote", int(alertas["lote"].eq(ultimo).sum()))
c2.metric("Alertas en total", len(alertas))
c3.metric("Clarificantes en el último lote", int(serie(df, "negocio", "clarificantes_semana").iloc[-1]),
          help="Presupuesto: 2 por semana")
costo = serie(df, "desempeno", "costo_por_lote_modelo")
c4.metric("Costo por lote del modelo (USD)", f"{costo.iloc[-1]:.2f}" if len(costo) else "sin etiquetas")

# --- Alertas ---
st.subheader("Alertas")
st.dataframe(alertas[["lote", "tipo", "metrica", "variable", "valor", "umbral", "detalle"]]
             .sort_values(["lote", "tipo"]), hide_index=True, use_container_width=True)

# --- Reentrenamientos ---
st.subheader("Reentrenamientos")
try:
    reent = cargar_reentrenamientos()
except Exception as e:  # MLflow apagado: el resto del tablero sigue funcionando
    reent = None
    st.info(f"No se pudo leer MLflow ({e}).")
if reent is not None:
    if reent.empty:
        st.caption("Todavía no hay reentrenamientos.")
    else:
        st.caption("Corridas con fase = reentrenamiento en MLflow. El lote es el lote de simulación "
                   "en el que el monitoreo pidió reentrenar (vacío = disparo manual o por calendario).")
        st.dataframe(reent.round(2), hide_index=True, use_container_width=True)

# --- Negocio y desempeño ---
izq, der = st.columns(2)
with izq:
    st.subheader("Negocio: clarificantes por semana")
    negocio = pd.DataFrame({"recomendados por el modelo": serie(df, "negocio", "clarificantes_semana")})
    negocio["presupuesto (k = 2)"] = 2
    st.line_chart(negocio)
with der:
    st.subheader("Desempeño: costo de errores por lote (USD)")
    st.caption("Con la merma real, que llega 2 semanas después (acumulado)")
    st.line_chart(pd.DataFrame({
        "modelo": serie(df, "desempeno", "costo_por_lote_modelo"),
        "regla por estilo": serie(df, "desempeno", "costo_por_lote_regla_estilo"),
        "sin acción (hoy)": serie(df, "desempeno", "costo_por_lote_sin_accion"),
    }))

izq, der = st.columns(2)
with izq:
    st.subheader("Desempeño: ROC-AUC acumulado y por lote")
    st.caption("Por lote: cada valor se ubica en el lote al que pertenece la merma real. Límite: media de la "
               "CV del PSet 3 − 2 desv. (0,389); 2 lotes seguidos bajo el límite disparan el reentrenamiento.")
    por_lote = df[(df["tipo"] == "desempeno") & (df["metrica"] == "roc_auc_lote")].copy()
    por_lote["lote_etiqueta"] = por_lote["variable"].str.extract(r"(\d+)").astype(float)
    st.line_chart(pd.DataFrame({
        "ROC-AUC acumulado": serie(df, "desempeno", "roc_auc"),
        "ROC-AUC del lote": por_lote.set_index("lote_etiqueta")["valor"].sort_index(),
        "límite (0,389)": 0.389,
    }))
with der:
    st.subheader("Modelo: probabilidad media y % con clarificante")
    st.line_chart(pd.DataFrame({
        "probabilidad media": serie(df, "modelo", "probabilidad_media"),
        "% con clarificante": serie(df, "modelo", "pct_aplicar"),
    }))

# --- Datos ---
st.subheader("Datos: deriva por variable (PSI contra el entrenamiento)")
st.caption("PSI > 0,25 con prueba KS/chi² significativa = alerta. coccion_mes y temporada_alta son de "
           "calendario: su cambio es esperado y no alerta.")
psi = (df[(df["tipo"] == "datos") & (df["metrica"] == "psi")]
       .pivot_table(index="lote", columns="variable", values="valor"))
st.line_chart(psi)
st.dataframe(psi.round(2), use_container_width=True)

# --- Sistema ---
izq, der = st.columns(2)
with izq:
    st.subheader("Sistema: latencia p95 (ms)")
    st.line_chart(pd.DataFrame({"latencia p95": serie(df, "sistema", "latencia_p95_ms"), "umbral": 500}))
with der:
    st.subheader("Sistema: errores, respaldo y faltantes (%)")
    st.caption("Errores 4xx (entrada inválida, umbral 5 %) y 5xx (error interno, umbral 0 %) sobre las "
               "solicitudes a /predecir de cada lote.")
    st.line_chart(pd.DataFrame({
        "% errores 4xx": serie(df, "sistema", "pct_errores_4xx") * 100,
        "% errores 5xx": serie(df, "sistema", "pct_errores_5xx") * 100,
        "% con respaldo": serie(df, "sistema", "pct_respaldo") * 100,
        "% faltantes": serie(df, "sistema", "pct_predicciones_faltantes") * 100,
    }))
st.subheader("Sistema: volumen por lote")
st.bar_chart(pd.DataFrame({
    "solicitudes": serie(df, "sistema", "volumen_solicitudes"),
    "predicciones": serie(df, "sistema", "volumen_predicciones"),
}))