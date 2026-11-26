"""
Reentrenamiento del modelo (sección 7 del PSet 4).

Qué hace:
1. Lee la OBT más reciente desde Snowflake. Solo trae órdenes cerradas, es decir, lotes que
   ya tienen la merma real (la etiqueta llega unas 2 semanas después de la cocción).
2. Separa una VENTANA DE EVALUACIÓN: las últimas N semanas de lotes con etiqueta (6 por defecto).
   Ni el champion ni el challenger vieron esos lotes al entrenar.
3. Entrena el CHALLENGER con todos los lotes ANTERIORES a la ventana, usando el MISMO pipeline
   y los MISMOS hiperparámetros del champion (se clona el pipeline del champion; solo cambian
   los datos).
4. Calcula el umbral del challenger con la misma regla del champion: k = 2 lotes por semana.
5. Compara ambos modelos en la ventana de evaluación: ROC-AUC, costo de negocio por lote y
   clarificantes por semana.
6. Registra el challenger en MLflow (nueva versión con alias @challenger) y deja una
   RECOMENDACIÓN. NO promueve nada: la promoción la decide una persona en Kestra (Pause).

Uso:
- Desde Kestra (flujo reentrenamiento): python -m src.reentrenar
- A mano (desde proyecto_final/, con MLflow encendido): python -m src.reentrenar --semanas-evaluacion 6
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import recall_score, roc_auc_score

from src import pipeline as P

RAIZ = Path(__file__).resolve().parents[1]   # proyecto_final/
try:                                          # en local lee el .env; en Kestra las variables ya vienen
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
except ImportError:
    pass

NOMBRE_MODELO = "merma-cerveza-modelo"
TABLA_OBT = "CERVECERIA_DB.OBT.OBT_MERMA_LOTES"
K_LOTES_SEMANA = 2          # presupuesto de clarificante: 2 lotes por semana
TOLERANCIA_ROC = 0.02       # el challenger puede perder hasta 0,02 de ROC-AUC si baja el costo


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def _llave_privada():
    from cryptography.hazmat.primitives import serialization
    ruta = Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    llave = serialization.load_pem_private_key(ruta.read_bytes(), password=None)
    return llave.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                               serialization.NoEncryption())


def _convertir_columna(serie):
    """Convierte los valores que entrega Snowflake (bool, Decimal, int) a tipos de pandas."""
    valores = serie.dropna()
    if len(valores) == 0:
        return serie
    if valores.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        return serie.map({True: True, False: False})          # se queda como True/False/NaN
    if valores.map(lambda v: isinstance(v, (int, float)) or type(v).__name__ == "Decimal").all():
        return pd.to_numeric(serie.astype(float))
    return serie


def leer_obt():
    """Lee la OBT de Snowflake (sin pyarrow, para que funcione con la imagen de la API)."""
    import snowflake.connector
    with snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        role=os.environ["SNOWFLAKE_ROLE"], warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"], private_key=_llave_privada(),
    ) as con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {TABLA_OBT}")
        df = pd.DataFrame(cur.fetchall(), columns=[c[0].lower() for c in cur.description])
    for col in df.columns:
        if col.startswith("fecha"):
            df[col] = pd.to_datetime(df[col])
        elif df[col].dtype == object:
            df[col] = _convertir_columna(df[col])
    df = df.sort_values(["fecha_coccion", "numero_documento"]).reset_index(drop=True)
    df[P.TARGET] = (df["merma_pct"] > 5.0).astype(int)   # misma definición del target (merma > 5 %)
    return df


def dividir_por_ventana(df, semanas):
    """Entrenamiento = todo lo anterior a las últimas `semanas` semanas; evaluación = esas semanas."""
    ultima_semana = df["fecha_coccion"].max().to_period("W").start_time
    inicio_eval = ultima_semana - pd.Timedelta(weeks=semanas - 1)
    entrenamiento = df[df["fecha_coccion"] < inicio_eval].reset_index(drop=True)
    evaluacion = df[df["fecha_coccion"] >= inicio_eval].reset_index(drop=True)
    return entrenamiento, evaluacion, inicio_eval


# ---------------------------------------------------------------------------
# Umbral y métricas
# ---------------------------------------------------------------------------
def umbral_k(modelo, entrenamiento):
    """Mismo criterio que preparar_api.py: umbral que deja en 'aplicar' a k lotes por semana."""
    X, _ = P.preparar_X_y(entrenamiento)
    lotes_semana = entrenamiento.groupby(entrenamiento["fecha_coccion"].dt.to_period("W")).size().mean()
    fraccion = min(K_LOTES_SEMANA / lotes_semana, 1.0)
    return float(np.quantile(modelo.predict_proba(X)[:, 1], 1 - fraccion)), fraccion


def evaluar(modelo, umbral, evaluacion):
    """Métricas técnicas y de negocio en la ventana de evaluación."""
    X, y = P.preparar_X_y(evaluacion)
    prob = modelo.predict_proba(X)[:, 1]
    aplicar = (prob >= umbral).astype(int)
    costo_fn, costo_fp = P.costos_por_lote(evaluacion)
    costo = ((aplicar == 0) & (y.values == 1)) * costo_fn.values + ((aplicar == 1) & (y.values == 0)) * costo_fp.values
    por_semana = pd.Series(aplicar).groupby(evaluacion["fecha_coccion"].dt.to_period("W").values).sum()
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() == 2 else float("nan"),
        "recall": float(recall_score(y, aplicar, zero_division=0)),
        "costo_por_lote": float(costo.sum() / len(y)),
        "clarificantes_semana_media": float(por_semana.mean()),
        "clarificantes_semana_max": int(por_semana.max()),
    }


def recomendar(champ, chall):
    """Regla de la recomendación (la decisión final es humana)."""
    mejor_costo = chall["costo_por_lote"] <= champ["costo_por_lote"]
    if np.isnan(chall["roc_auc"]) or np.isnan(champ["roc_auc"]):
        roc_ok = True                     # sin las dos clases en la ventana no se puede comparar ROC-AUC
    else:
        roc_ok = chall["roc_auc"] >= champ["roc_auc"] - TOLERANCIA_ROC
    if mejor_costo and roc_ok:
        return "PROMOVER", "el challenger tiene menor o igual costo y no pierde ROC-AUC"
    if not mejor_costo:
        return "MANTENER", "el challenger cuesta más por lote que el champion"
    return "MANTENER", f"el challenger pierde más de {TOLERANCIA_ROC} de ROC-AUC"


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semanas-evaluacion", type=int, default=6)
    args = parser.parse_args()

    mlflow = P.configurar_mlflow()
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    client = MlflowClient()

    # 1) Champion actual (el número de versión, porque get_model_info no acepta alias)
    v_champ = client.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    champion = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}/{v_champ.version}")
    umbral_champ = float(v_champ.tags.get("umbral_decision", "0.5"))

    # 2) Datos recientes y ventana de evaluación
    df = P.agregar_features(leer_obt())
    entrenamiento, evaluacion, inicio_eval = dividir_por_ventana(df, args.semanas_evaluacion)
    X_tr, y_tr = P.preparar_X_y(entrenamiento)

    # 3) Challenger: mismo pipeline y mismos hiperparámetros, datos más recientes
    challenger = clone(champion).fit(X_tr, y_tr)
    umbral_chall, fraccion = umbral_k(challenger, entrenamiento)

    # 4) Comparación en la ventana que ninguno de los dos vio
    m_champ = evaluar(champion, umbral_champ, evaluacion)
    m_chall = evaluar(challenger, umbral_chall, evaluacion)
    recomendacion, motivo = recomendar(m_champ, m_chall)

    # 5) Registro en MLflow
    fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    hiper = {k: v for k, v in challenger.named_steps["modelo"].get_params().items()
             if isinstance(v, (int, float, str, bool, type(None)))}
    with mlflow.start_run(run_name=f"reentrenamiento {fecha}") as run:
        mlflow.log_params({
            **hiper,
            "semilla": P.SEMILLA,
            "champion_version": v_champ.version,
            "fecha_corte_entrenamiento": str(entrenamiento["fecha_coccion"].max().date()),
            "inicio_ventana_evaluacion": str(inicio_eval.date()),
            "fin_ventana_evaluacion": str(evaluacion["fecha_coccion"].max().date()),
            "semanas_evaluacion": args.semanas_evaluacion,
            "n_entrenamiento": len(entrenamiento),
            "n_evaluacion": len(evaluacion),
            "umbral_decision": round(umbral_chall, 6),
        })
        mlflow.log_metrics({**{f"challenger_{k}": v for k, v in m_chall.items() if not pd.isna(v)},
                            **{f"champion_{k}": v for k, v in m_champ.items() if not pd.isna(v)}})
        mlflow.set_tags({"fase": "reentrenamiento", "familia": type(challenger.named_steps["modelo"]).__name__,
                         "commit_git": P.commit_git(), "recomendacion": recomendacion})
        mlflow.sklearn.log_model(
            challenger, "modelo",
            signature=infer_signature(X_tr, challenger.predict(X_tr)),
            pip_requirements=["scikit-learn==1.5.2", "pandas==2.2.3", "numpy==1.26.4"],
        )
        run_id = run.info.run_id

    nueva = mlflow.register_model(f"runs:/{run_id}/modelo", NOMBRE_MODELO)
    etiquetas = {
        "umbral_decision": f"{umbral_chall:.6f}",
        "regla_decision": f"top-{K_LOTES_SEMANA} lotes por semana (~{fraccion:.0%} de los lotes)",
        "origen": "reentrenamiento",
        "comparado_con": f"versión {v_champ.version} (champion)",
        "recomendacion": recomendacion,
    }
    for clave, valor in etiquetas.items():
        client.set_model_version_tag(NOMBRE_MODELO, nueva.version, clave, valor)
    client.set_registered_model_alias(NOMBRE_MODELO, "challenger", nueva.version)

    # 6) Resumen (log) y salidas para Kestra
    tabla = pd.DataFrame({f"champion v{v_champ.version}": m_champ,
                          f"challenger v{nueva.version}": m_chall}).round(4)
    print(f"Entrenamiento del challenger: {len(entrenamiento)} lotes hasta "
          f"{entrenamiento['fecha_coccion'].max().date()}")
    print(f"Ventana de evaluación: {len(evaluacion)} lotes desde {inicio_eval.date()}")
    print(tabla.to_string())
    print(f"Recomendación: {recomendacion} ({motivo})")

    salidas = {
        "champion_version": str(v_champ.version),
        "challenger_version": str(nueva.version),
        "recomendacion": recomendacion,
        "motivo": motivo,
        "comparacion": tabla.to_string(),
    }
    print("::" + json.dumps({"outputs": salidas}) + "::")


if __name__ == "__main__":
    main()