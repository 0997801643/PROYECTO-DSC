"""
Tablero de monitoreo (sección 6.2 del PSet 4).

Lee de Snowflake la tabla PRODUCCION.MONITOREO (la llena el flujo de Kestra "monitoreo") y
muestra, lote por lote: alertas, negocio, desempeño, modelo, datos y sistema.

Se abre en http://localhost:8501 (servicio "tablero" del docker-compose).
"""
import os
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
    st.subheader("Desempeño: ROC-AUC acumulado")
    st.line_chart(pd.DataFrame({"ROC-AUC": serie(df, "desempeno", "roc_auc"), "azar (0,5)": 0.5}))
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
    st.subheader("Sistema: respaldo y predicciones faltantes (%)")
    st.line_chart(pd.DataFrame({
        "% con respaldo": serie(df, "sistema", "pct_respaldo") * 100,
        "% faltantes": serie(df, "sistema", "pct_predicciones_faltantes") * 100,
    }))