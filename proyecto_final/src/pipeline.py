"""
Preparación de los datos para el modelo de merma alta.

- agregar_features(): crea las features derivadas (sin fuga de datos).
- preparar_X_y():      separa las features (X) del target (y).
- dividir_train_test(): hold-out temporal (el último 20 % de lotes por fecha de cocción).
- construir_pipeline(): Pipeline = ColumnTransformer (imputación, codificación, escalado) + modelo.

La imputación, codificación y escalado viven DENTRO del Pipeline, así se ajustan
solo con los datos de entrenamiento de cada pliegue.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "merma_alta"
SEMILLA = 42

# Meses con mayor proporción de lotes con merma alta en la EDA (temporada alta)
MESES_TEMPORADA_ALTA = [4, 5, 6, 8]
VENTANA_CARGA_DIAS = 14

# Las mediciones del microscopio (viabilidad, células, litros) existen solo desde agosto:
# con validación temporal, los pliegues de entrenamiento no tienen ningún valor -> se excluyen
# y queda solo la bandera con_microscopio (ver hallazgo 2 de la EDA).
FEATURES_MICRO_EXCLUIDAS = ["micro_viabilidad_pct", "micro_celulas_viables_m_ml", "micro_litros_levadura"]

FEATURES_NUM = [
    "generacion_levadura",
    "n_cocciones",
    "coccion_mes",
    "coccion_dia_semana",
    "carga_planta_14d",        # derivada
]
FEATURES_BIN = [
    "levadura_nueva",
    "sin_generacion",
    "con_microscopio",
    "temporada_alta",          # derivada
]
FEATURES_CAT = ["cepa"]
FEATURES = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT


def _a_binaria(serie):
    """True/False/None (o texto) -> 1.0 / 0.0 / NaN."""
    mapa = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}
    return serie.map(mapa).astype(float)


def agregar_features(df):
    """Agrega las features derivadas. Solo usa información conocida el día de la cocción."""
    df = df.sort_values(["fecha_coccion", "numero_documento"]).reset_index(drop=True).copy()

    # 1) Carga de planta: lotes cocinados en los 14 días ANTERIORES a la cocción de este lote.
    #    Solo usa fechas de cocción pasadas (no usa la merma de nadie) -> sin fuga.
    fechas = df["fecha_coccion"].values.astype("datetime64[ns]")
    inicio = fechas - np.timedelta64(VENTANA_CARGA_DIAS, "D")
    df["carga_planta_14d"] = np.searchsorted(fechas, fechas, side="left") - np.searchsorted(fechas, inicio, side="left")

    # 2) Temporada alta: regla fija definida con la EDA y el conocimiento de planta
    df["temporada_alta"] = df["coccion_mes"].isin(MESES_TEMPORADA_ALTA).astype(float)

    # Banderas como 0/1
    for c in ["levadura_nueva", "sin_generacion", "con_microscopio"]:
        df[c] = _a_binaria(df[c])
    # La levadura nueva se define por la generación (0); si la generación falta, queda NaN
    df["levadura_nueva"] = np.where(df["generacion_levadura"].isna(), np.nan,
                                    (df["generacion_levadura"] == 0).astype(float))
    return df


def preparar_X_y(df):
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()


def dividir_train_test(df, frac_test=0.2):
    """Hold-out temporal: los lotes más recientes quedan para la prueba final."""
    df = df.sort_values(["fecha_coccion", "numero_documento"]).reset_index(drop=True)
    corte = int(len(df) * (1 - frac_test))
    return df.iloc[:corte].reset_index(drop=True), df.iloc[corte:].reset_index(drop=True)


def esquema_cv(n_splits=5):
    """Validación cruzada temporal: siempre se entrena con el pasado y se valida con el futuro."""
    return TimeSeriesSplit(n_splits=n_splits)


def construir_preprocesador(escalar=True):
    num = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        num.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), FEATURES_NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), FEATURES_BIN),
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), FEATURES_CAT),
    ])


def construir_pipeline(modelo, escalar=True):
    """Pipeline completo: preparación + modelo. Se ajusta como un solo objeto."""
    return Pipeline([("prep", construir_preprocesador(escalar)), ("modelo", modelo)])

# ---------------------------------------------------------------------------
# Métricas y registro en MLflow
# ---------------------------------------------------------------------------
import os
import subprocess
import time

from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate

# Costos de la planta
PRECIO_LITRO = {   # USD por litro de cerveza, según el estilo
    "American Lager": 0.427, "Oktoberfest": 0.427, "Lagarto": 0.540, "Red Llama": 0.509,
    "Condor": 0.582, "Tortuga": 0.665, "Iguana": 0.738, "Belgian Tripel": 0.743, "Honey Bock": 0.786,
}
COSTO_CLARIF_LITRO = 57.12 / 4 * 0.5 / 1000   # USD/L: $57.12 por 4 kg, 0.5 kg por cada 1000 L
META_MERMA = 0.05                             # con clarificante, la merma baja a la meta (5 %)


def costos_por_lote(df):
    """Costo de cada tipo de error para cada lote (USD)."""
    exceso_litros = (df["merma_litros"] - META_MERMA * df["cantidad_planificada"]).clip(lower=0)
    costo_fn = exceso_litros * df["estilo"].map(PRECIO_LITRO)          # merma alta sin clarificante
    costo_fp = COSTO_CLARIF_LITRO * df["cantidad_planificada"]          # clarificante sin necesidad
    return costo_fn, costo_fp


# Costo promedio por lote en el entrenamiento (calculado en 03_baseline_candidatos con costos_por_lote)
COSTO_FN = 108.07   # lote con merma alta SIN clarificante
COSTO_FP = 36.80    # lote normal CON clarificante



def costo_negocio(y_true, y_pred):
    """Costo promedio por lote (USD) de los errores del modelo."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (fn * COSTO_FN + fp * COSTO_FP) / len(y_true)


METRICA_TECNICA = "roc_auc"
SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "costo": make_scorer(costo_negocio, greater_is_better=False),
}


def commit_git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "sin_git"


def configurar_mlflow():
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "merma-cerveza"))
    return mlflow


def evaluar_cv(pipe, X, y):
    """Validación cruzada temporal: métricas de entrenamiento y de validación por pliegue."""
    t0 = time.time()
    cv = cross_validate(pipe, X, y, cv=esquema_cv(), scoring=SCORING,
                        return_train_score=True, error_score="raise")
    res = {"segundos_cv": time.time() - t0}
    for m in SCORING:
        signo = -1 if m == "costo" else 1          # el costo se guarda en USD positivos
        tr, va = signo * cv[f"train_{m}"], signo * cv[f"test_{m}"]
        res[f"train_{m}"] = float(tr.mean())
        res[f"cv_{m}_media"] = float(va.mean())
        res[f"cv_{m}_std"] = float(va.std())
    res["brecha"] = res[f"train_{METRICA_TECNICA}"] - res[f"cv_{METRICA_TECNICA}_media"]
    return res


def evaluar_y_registrar(nombre, modelo, X, y, fase, familia, meta_datos,
                        escalar=True, nested=False, guardar_modelo=True, params_extra=None):
    """Evalúa un modelo con la CV temporal y registra TODO en MLflow. Devuelve (run_id, métricas, pipeline)."""
    import mlflow
    from mlflow.models import infer_signature

    pipe = construir_pipeline(modelo, escalar)
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        metricas = evaluar_cv(pipe, X, y)
        t0 = time.time()
        pipe.fit(X, y)
        metricas["segundos_entrenamiento"] = time.time() - t0

        params = {k: v for k, v in modelo.get_params().items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        mlflow.log_params({**params, "semilla": SEMILLA, **(params_extra or {}), **meta_datos})
        mlflow.log_metrics(metricas)
        mlflow.set_tags({"fase": fase, "familia": familia, "commit_git": commit_git(),
                         "esquema_cv": "TimeSeriesSplit(5)", "metrica_tecnica": METRICA_TECNICA})
        if guardar_modelo:
            ejemplo = X.dropna().head(5) if len(X.dropna()) else X.head(5)
            firma = infer_signature(X, pipe.predict(X))
            mlflow.sklearn.log_model(pipe, "modelo", signature=firma, input_example=ejemplo)
        return run.info.run_id, metricas, pipe