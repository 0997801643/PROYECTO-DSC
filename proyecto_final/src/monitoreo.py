"""
Monitoreo del modelo en producción (sección 6 del PSet 4).

Para cada lote de simulación (una semana de cocción) calcula y guarda en Snowflake
(tabla PRODUCCION.MONITOREO) cuatro grupos de métricas:

- DATOS:     % de nulos, % fuera del rango de entrenamiento, categorías no vistas,
             PSI y pruebas estadísticas (KS para numéricas, chi² para categóricas)
             contra la distribución del entrenamiento.
- MODELO:    probabilidad media, % de lotes con "aplicar clarificante" y PSI de las
             probabilidades contra las del entrenamiento.
- DESEMPEÑO: con las etiquetas reales (la merma) que llegan con RETRASO_ETIQUETAS semanas:
             ROC-AUC, recall, precisión y costo por lote (modelo vs. regla por estilo vs. no hacer nada).
             Además, ROC-AUC POR LOTE contra la CV del champion (PSet 3): si cae bajo
             media − 2 desv. en 2 lotes etiquetados seguidos, se pide un reentrenamiento.
- SISTEMA:   volumen de predicciones, latencia p50 y p95, % de respuestas con el respaldo y
             % de predicciones faltantes (errores).
- NEGOCIO:   clarificantes recomendados en la semana vs. el presupuesto (k = 2).

Si una métrica supera su umbral, se marca ALERTA y se informa a Kestra (n_alertas, alertas,
disparar_reentrenamiento y motivo_reentrenamiento).

Uso:
  En Kestra (contenedor api-merma:local):   python monitoreo.py --lotes todos
  Local, una sola vez, para crear la referencia del entrenamiento (necesita MLflow):
      python -m src.monitoreo --crear-referencia
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]                       # proyecto_final/
RUTA_SIM = RAIZ / "data" / "simulacion" / "lotes_simulacion.csv"
RUTA_REF = RAIZ / "data" / "monitoreo" / "referencia.csv"
RUTA_LINEA_BASE = RAIZ / "api" / "linea_base.json"

TABLA_PRED = "PRODUCCION.PREDICCIONES"
TABLA_MON = "PRODUCCION.MONITOREO"

NUMERICAS = ["generacion_levadura", "n_cocciones", "coccion_mes", "coccion_dia_semana", "carga_planta_14d"]
CATEGORICAS = ["levadura_nueva", "sin_generacion", "con_microscopio", "temporada_alta", "cepa"]
# Variables de calendario: en producción SIEMPRE llegan meses posteriores al entrenamiento, así que
# su cambio de distribución es esperado. Se miden y se guardan, pero no disparan alerta.
CALENDARIO = ["coccion_mes", "temporada_alta"]
# Variables con peso en el modelo en producción (coeficiente distinto de 0 en la logística L1 v1):
# solo su deriva puede cambiar las predicciones. La deriva de las demás (por ejemplo con_microscopio,
# que cambió porque el microscopio se compró en agosto) se mide y se guarda, pero no alerta.
# La calidad de datos (nulos, fuera de rango, cepas nuevas) sí alerta en todas las variables.
CON_PESO_EN_MODELO = ["generacion_levadura", "temporada_alta"]

K_LOTES_SEMANA = 2          # presupuesto de clarificante (lotes por semana)
RETRASO_ETIQUETAS = 2       # la merma real se conoce 2 semanas después de la cocción

# Deriva conceptual: ROC-AUC por lote contra la CV del champion en el PSet 3
# (corrida de tuning logistica_cfg10, la configuración de la versión 1).
CV_ROC_MEDIA, CV_ROC_STD = 0.635, 0.123
LIMITE_ROC_LOTE = CV_ROC_MEDIA - 2 * CV_ROC_STD      # 0,389
N_LOTES_SEGUIDOS = 2        # lotes etiquetados seguidos bajo el límite para pedir reentrenamiento

# Umbrales de alerta
UMBRAL = {
    "psi": 0.25,                 # PSI > 0,25 = cambio importante de distribución...
    "p_valor": 0.05,             # ...confirmado con KS / chi² (p < 0,05)
    "nulos": 0.20,               # más del 20 % de nulos en una variable
    "fuera_rango": 0.20,         # más del 20 % de valores fuera del rango del entrenamiento
    "categoria_nueva": 0.0,      # cualquier cepa no vista
    "roc_auc_min": 0.50,         # peor que el azar
    "latencia_p95_ms": 500.0,
    "respaldo": 0.0,             # cualquier respuesta con el respaldo
    "faltantes": 0.0,            # cualquier predicción que no llegó
}

# Costos de la planta (los mismos de src/pipeline.py, PSet 3)
PRECIO_LITRO = {
    "American Lager": 0.427, "Oktoberfest": 0.427, "Lagarto": 0.540, "Red Llama": 0.509,
    "Condor": 0.582, "Tortuga": 0.665, "Iguana": 0.738, "Belgian Tripel": 0.743, "Honey Bock": 0.786,
}
COSTO_CLARIF_LITRO = 57.12 / 4 * 0.5 / 1000
META_MERMA = 0.05


# ---------------------------------------------------------------------------
# Pruebas de deriva
# ---------------------------------------------------------------------------
def _como_texto(serie):
    """Normaliza categorías: 1, 1.0 y "1" deben ser la misma categoría (la referencia viene de un CSV
    con decimales y la producción de un JSON con enteros)."""
    serie = pd.Series(serie).dropna()
    numeros = pd.to_numeric(serie, errors="coerce")
    if numeros.notna().all():
        return numeros.map(lambda x: f"{x:g}")
    return serie.astype(str)


def psi(ref, act, categorica, bins=10):
    """Population Stability Index: < 0,1 estable · 0,1–0,25 moderado · > 0,25 cambio importante."""
    ref, act = pd.Series(ref).dropna(), pd.Series(act).dropna()
    if len(act) == 0:
        return np.nan
    if not categorica:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(cortes) < 3:
            categorica = True                 # pocas categorías: se trata como categórica
    if categorica:
        ref, act = _como_texto(ref), _como_texto(act)
        cats = sorted(set(ref) | set(act))
        p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
        q = act.value_counts(normalize=True).reindex(cats, fill_value=0).values
    else:
        cortes[0], cortes[-1] = -np.inf, np.inf
        p = np.histogram(ref, cortes)[0] / len(ref)
        q = np.histogram(act, cortes)[0] / len(act)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def p_valor_ks(ref, act):
    ref, act = pd.Series(ref).dropna(), pd.Series(act).dropna()
    return float(stats.ks_2samp(ref, act).pvalue) if len(act) >= 2 else np.nan


def p_valor_chi2(ref, act):
    ref, act = _como_texto(ref), _como_texto(act)
    cats = sorted(set(ref) | set(act))
    tabla = np.array([ref.value_counts().reindex(cats, fill_value=0).values,
                      act.value_counts().reindex(cats, fill_value=0).values])
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2 or len(act) < 2:
        return np.nan
    return float(stats.chi2_contingency(tabla)[1])


# ---------------------------------------------------------------------------
# Costo de negocio (igual que el PSet 3)
# ---------------------------------------------------------------------------
def costo_por_lote(lotes, aplicar):
    """Costo promedio por lote (USD) de los errores: merma alta sin clarificante + clarificante sin necesidad."""
    y = lotes["merma_alta"].astype(int).values
    exceso = (lotes["merma_litros"] - META_MERMA * lotes["cantidad_planificada"]).clip(lower=0)
    costo_fn = (exceso * lotes["estilo"].map(PRECIO_LITRO)).values
    costo_fp = (COSTO_CLARIF_LITRO * lotes["cantidad_planificada"]).values
    aplicar = np.asarray(aplicar).astype(int)
    total = ((aplicar == 0) & (y == 1)) * costo_fn + ((aplicar == 1) & (y == 0)) * costo_fp
    return float(total.sum() / len(y))


# ---------------------------------------------------------------------------
# Snowflake
# ---------------------------------------------------------------------------
def conectar():
    import snowflake.connector
    from cryptography.hazmat.primitives import serialization
    llave = serialization.load_pem_private_key(
        Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"]).read_bytes(), password=None)
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        role=os.environ["SNOWFLAKE_ROLE"], warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        private_key=llave.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))


def leer_predicciones(con):
    """Última predicción de cada lote de cerveza en cada lote de simulación."""
    cur = con.cursor()
    cur.execute(f"""
        SELECT LOTE_SIMULACION, NUMERO_DOCUMENTO, ENTRADAS, PROBABILIDAD, DECISION,
               VERSION_MODELO, RESPALDO, LATENCIA_MS
        FROM {TABLA_PRED}
        WHERE LOTE_SIMULACION IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY LOTE_SIMULACION, NUMERO_DOCUMENTO
                                   ORDER BY TIMESTAMP_PREDICCION DESC) = 1""")
    columnas = [c[0].lower() for c in cur.description]
    pred = pd.DataFrame(cur.fetchall(), columns=columnas)
    entradas = pd.DataFrame([json.loads(e) if isinstance(e, str) else e for e in pred["entradas"]])
    pred = pd.concat([pred.drop(columns="entradas"),
                      entradas[NUMERICAS + CATEGORICAS + ["estilo"]]], axis=1)
    pred["numero_documento"] = pred["numero_documento"].astype(str)
    return pred


def guardar(con, filas, lotes):
    cur = con.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLA_MON} (
            FECHA_EJECUCION TIMESTAMP_TZ, LOTE_SIMULACION INTEGER, TIPO VARCHAR, METRICA VARCHAR,
            VARIABLE VARCHAR, VALOR FLOAT, UMBRAL FLOAT, ALERTA BOOLEAN, DETALLE VARCHAR)""")
    cur.execute(f"DELETE FROM {TABLA_MON} WHERE LOTE_SIMULACION IN ({','.join(map(str, lotes))})")
    ahora = datetime.now(timezone.utc).isoformat()
    # Los números se envían redondeados y con tipo explícito (::FLOAT): en una inserción de muchas
    # filas Snowflake deduce el tipo de la columna con la primera fila, y un entero ahí hacía fallar
    # los decimales siguientes.
    cur.executemany(
        f"INSERT INTO {TABLA_MON} VALUES (%s, %s, %s, %s, %s, %s::FLOAT, %s::FLOAT, %s, %s)",
        [(ahora, int(f["lote"]), f["tipo"], f["metrica"], f["variable"],
          None if f["valor"] is None or pd.isna(f["valor"]) else round(float(f["valor"]), 6),
          None if f["umbral"] is None else float(f["umbral"]), bool(f["alerta"]), f["detalle"])
         for f in filas])


# ---------------------------------------------------------------------------
# Métrica técnica por lote y regla de reentrenamiento
# ---------------------------------------------------------------------------
def roc_auc_lote(pred, sim, lote):
    """ROC-AUC de UN lote de simulación con su merma real (NaN si el lote no tiene las dos clases)."""
    datos = (pred[pred["lote_simulacion"] == lote]
             .merge(sim[["lote_simulacion", "numero_documento", "merma_alta"]],
                    on=["lote_simulacion", "numero_documento"]))
    y = datos["merma_alta"].astype(int)
    if y.nunique() < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, datos["probabilidad"]))


def regla_reentrenamiento(pred, sim, ultimo_lote):
    """Pide reentrenar si los últimos N lotes CON etiqueta (y con ROC-AUC calculable) quedaron
    bajo el límite de la CV. La deriva de datos sola no dispara: tiene que caer el desempeño."""
    hasta = ultimo_lote - RETRASO_ETIQUETAS
    rocs = [(l, roc_auc_lote(pred, sim, l)) for l in range(1, hasta + 1)]
    rocs = [(l, r) for l, r in rocs if not np.isnan(r)]
    ultimos = rocs[-N_LOTES_SEGUIDOS:]
    if len(ultimos) == N_LOTES_SEGUIDOS and all(r < LIMITE_ROC_LOTE for _, r in ultimos):
        detalle = ", ".join(f"lote {l}: {r:.2f}" for l, r in ultimos)
        return True, (f"desempeño: ROC-AUC bajo {LIMITE_ROC_LOTE:.3f} (media CV − 2 desv.) "
                      f"en {N_LOTES_SEGUIDOS} lotes seguidos ({detalle}), detectado en el lote {ultimo_lote}")
    return False, ""


# ---------------------------------------------------------------------------
# Métricas de un lote
# ---------------------------------------------------------------------------
def metricas_lote(lote, pred, ref, sim, linea_base):
    filas = []

    def agregar(tipo, metrica, valor, umbral=None, alerta=False, variable="", detalle=""):
        filas.append({"lote": lote, "tipo": tipo, "metrica": metrica, "variable": variable,
                      "valor": valor, "umbral": umbral, "alerta": bool(alerta), "detalle": detalle})

    act = pred[pred["lote_simulacion"] == lote]
    esperadas = sim[sim["lote_simulacion"] == lote]

    # --- DATOS ---
    for v in NUMERICAS + CATEGORICAS:
        categorica = v in CATEGORICAS
        nulos = act[v].isna().mean()
        agregar("datos", "pct_nulos", nulos, UMBRAL["nulos"], nulos > UMBRAL["nulos"], v)
        # Con ~5 lotes por semana el PSI solo es muy inestable: se alerta cuando el PSI es alto
        # Y la prueba estadística (que sí considera el tamaño de muestra) confirma el cambio.
        valor_psi = psi(ref[v], act[v], categorica)
        p = p_valor_chi2(ref[v], act[v]) if categorica else p_valor_ks(ref[v], act[v])
        cambio = valor_psi > UMBRAL["psi"] and p < UMBRAL["p_valor"]
        if cambio and v in CALENDARIO:
            agregar("datos", "psi", valor_psi, UMBRAL["psi"], False, v,
                    f"p = {p:.4f}: cambio esperado (variable de calendario, no alerta)")
        elif cambio and v not in CON_PESO_EN_MODELO:
            agregar("datos", "psi", valor_psi, UMBRAL["psi"], False, v,
                    f"p = {p:.4f}: cambio confirmado, pero la variable no tiene peso en el modelo (no alerta)")
        else:
            agregar("datos", "psi", valor_psi, UMBRAL["psi"], cambio, v,
                    f"p = {p:.4f}: cambio de distribución confirmado" if cambio else f"p = {p:.4f}")
        agregar("datos", "p_valor_chi2" if categorica else "p_valor_ks", p, UMBRAL["p_valor"], False, v)
        if not categorica:
            fuera = ((act[v] < ref[v].min()) | (act[v] > ref[v].max())).mean()
            agregar("datos", "pct_fuera_rango", fuera, UMBRAL["fuera_rango"],
                    fuera > UMBRAL["fuera_rango"] and v not in CALENDARIO, v,
                    f"rango del entrenamiento [{ref[v].min():g}, {ref[v].max():g}]")
    nuevas = (~act["cepa"].isin(ref["cepa"].dropna().unique())).mean()
    agregar("datos", "pct_categoria_nueva", nuevas, UMBRAL["categoria_nueva"],
            nuevas > UMBRAL["categoria_nueva"], "cepa")

    # --- MODELO ---
    agregar("modelo", "probabilidad_media", act["probabilidad"].mean())
    agregar("modelo", "pct_aplicar", (act["decision"] == "aplicar_clarificante").mean())
    psi_prob = psi(ref["prob"], act["probabilidad"], categorica=False)
    p_prob = p_valor_ks(ref["prob"], act["probabilidad"])
    cambio = psi_prob > UMBRAL["psi"] and p_prob < UMBRAL["p_valor"]
    agregar("modelo", "psi_probabilidad", psi_prob, UMBRAL["psi"], cambio, "probabilidad", f"p = {p_prob:.4f}")

    # --- DESEMPEÑO (etiquetas con retraso: lotes <= lote - RETRASO_ETIQUETAS) ---
    hasta = lote - RETRASO_ETIQUETAS
    con_etiqueta = (pred[pred["lote_simulacion"] <= hasta]
                    .merge(sim[["lote_simulacion", "numero_documento", "merma_alta", "merma_litros",
                                "cantidad_planificada"]], on=["lote_simulacion", "numero_documento"]))
    if len(con_etiqueta):
        y = con_etiqueta["merma_alta"].astype(int)
        aplicar = (con_etiqueta["decision"] == "aplicar_clarificante").astype(int)
        detalle = f"acumulado de los lotes 1 a {hasta} ({len(con_etiqueta)} lotes de cerveza)"
        if y.nunique() == 2:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, con_etiqueta["probabilidad"])
            agregar("desempeno", "roc_auc", auc, UMBRAL["roc_auc_min"], auc < UMBRAL["roc_auc_min"],
                    detalle=detalle)
        agregar("desempeno", "recall", (aplicar[y == 1] == 1).mean() if (y == 1).any() else np.nan, detalle=detalle)
        agregar("desempeno", "precision", (y[aplicar == 1] == 1).mean() if (aplicar == 1).any() else np.nan,
                detalle=detalle)
        c_modelo = costo_por_lote(con_etiqueta, aplicar)
        c_regla = costo_por_lote(con_etiqueta, con_etiqueta["estilo"].isin(linea_base["estilos_alerta"]))
        c_nada = costo_por_lote(con_etiqueta, np.zeros(len(con_etiqueta)))
        agregar("desempeno", "costo_por_lote_modelo", c_modelo, c_regla, c_modelo > c_regla, detalle=detalle)
        agregar("desempeno", "costo_por_lote_regla_estilo", c_regla, detalle=detalle)
        agregar("desempeno", "costo_por_lote_sin_accion", c_nada, detalle=detalle)
        # Métrica técnica DEL LOTE que acaba de recibir su etiqueta, contra la CV del PSet 3
        roc_lote = roc_auc_lote(pred, sim, hasta)
        if not np.isnan(roc_lote):
            # El valor es del lote `hasta` (su merma llegó ahora), no del lote que se monitorea:
            # se guarda con variable = "lote N" para que quede claro a qué lote pertenece.
            agregar("desempeno", "roc_auc_lote", roc_lote, LIMITE_ROC_LOTE, roc_lote < LIMITE_ROC_LOTE,
                    variable=f"lote {hasta}",
                    detalle=f"ROC-AUC del lote {hasta} (su merma real se conoció al monitorear el lote {lote}); "
                            f"CV del PSet 3: {CV_ROC_MEDIA} ± {CV_ROC_STD}, límite media − 2 desv.")
    else:
        agregar("desempeno", "sin_etiquetas", None,
                detalle=f"la merma real llega {RETRASO_ETIQUETAS} semanas después")

    # --- SISTEMA ---
    agregar("sistema", "volumen_predicciones", len(act), detalle=f"{len(esperadas)} lotes de cerveza esperados")
    agregar("sistema", "latencia_p50_ms", act["latencia_ms"].quantile(0.50))
    p95 = act["latencia_ms"].quantile(0.95)
    agregar("sistema", "latencia_p95_ms", p95, UMBRAL["latencia_p95_ms"], p95 > UMBRAL["latencia_p95_ms"])
    respaldo = act["respaldo"].astype(bool).mean()
    agregar("sistema", "pct_respaldo", respaldo, UMBRAL["respaldo"], respaldo > UMBRAL["respaldo"])
    faltantes = 1 - len(act) / max(len(esperadas), 1)
    agregar("sistema", "pct_predicciones_faltantes", faltantes, UMBRAL["faltantes"],
            faltantes > UMBRAL["faltantes"])

    # --- NEGOCIO ---
    n_aplicar = int((act["decision"] == "aplicar_clarificante").sum())
    agregar("negocio", "clarificantes_semana", n_aplicar, K_LOTES_SEMANA, n_aplicar > K_LOTES_SEMANA,
            detalle=f"presupuesto: {K_LOTES_SEMANA} por semana")
    return filas


# ---------------------------------------------------------------------------
# Referencia del entrenamiento (se corre una vez, en local)
# ---------------------------------------------------------------------------
def crear_referencia():
    import mlflow
    from mlflow import MlflowClient
    sys.path.insert(0, str(RAIZ))
    from src import pipeline as P
    from src.datos import cargar_obt

    P.configurar_mlflow()
    version = MlflowClient().get_model_version_by_alias("merma-cerveza-modelo", "champion")
    modelo = mlflow.sklearn.load_model(f"models:/merma-cerveza-modelo/{version.version}")
    train, _ = P.dividir_train_test(P.agregar_features(cargar_obt()))
    X, y = P.preparar_X_y(train)
    ref = X.copy()
    ref["merma_alta"] = y.values
    ref["prob"] = modelo.predict_proba(X)[:, 1]
    RUTA_REF.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(RUTA_REF, index=False)
    print(f"Referencia guardada: {len(ref)} lotes del entrenamiento (champion versión {version.version})")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lotes", default="todos")
    parser.add_argument("--crear-referencia", action="store_true")
    args = parser.parse_args()
    if args.crear_referencia:
        return crear_referencia()

    ref = pd.read_csv(RUTA_REF)
    sim = pd.read_csv(RUTA_SIM, dtype={"numero_documento": str})
    linea_base = json.loads(RUTA_LINEA_BASE.read_text(encoding="utf-8"))

    with conectar() as con:
        pred = leer_predicciones(con)
        lotes = ([int(x) for x in sorted(pred["lote_simulacion"].unique())] if args.lotes == "todos"
                 else [int(x) for x in args.lotes.split(",")])
        filas = [f for lote in lotes for f in metricas_lote(lote, pred, ref, sim, linea_base)]
        guardar(con, filas, lotes)

    disparar, motivo = regla_reentrenamiento(pred, sim, max(lotes))
    alertas = [f for f in filas if f["alerta"]]
    print(f"{len(filas)} métricas guardadas en {TABLA_MON} para los lotes {lotes}")
    for f in alertas:
        print(f"ALERTA | lote {f['lote']} | {f['tipo']} | {f['metrica']} {f['variable']} = "
              f"{f['valor']:.3f} (umbral {f['umbral']}) {f['detalle']}")
    resumen = "; ".join(f"lote {f['lote']}: {f['metrica']} {f['variable']}".strip() for f in alertas)
    # Salidas para Kestra (outputs.<tarea>.vars)
    if disparar:
        print(f"REENTRENAMIENTO SOLICITADO | {motivo}")
    print("::" + json.dumps({"outputs": {"n_alertas": len(alertas), "alertas": resumen or "ninguna",
                                         "disparar_reentrenamiento": "si" if disparar else "no",
                                         "motivo_reentrenamiento": motivo or "ninguno"}}) + "::")


if __name__ == "__main__":
    main()
