"""
Números finales para el documento técnico (secciones 6.2, 8.1 y 9 del PSet 4).

1. Lote por lote (sección 9): costo de los errores por lote (USD) de cada política
   (sin acción, regla por estilo, modelo umbral fijo, modelo top-2 por semana), ROC-AUC y
   clarificantes usados.
2. Deriva conceptual (sección 6.2): ROC-AUC y costo de cada lote contra el promedio y la
   desviación estándar de la validación cruzada del PSet 3 (corrida del champion en MLflow).
3. Subgrupos (sección 8.1): métrica técnica, recall, tasa de "aplicar" y costo por cepa de
   levadura y por uso de microscopio, en las semanas normales (1-9).

Usa el champion guardado en tests/fixtures/modelo (versión 1).
Salida: tablas en pantalla y data/simulacion/por_lote.csv, data/simulacion/subgrupos.csv
Uso (desde proyecto_final/, con MLflow encendido para leer la CV):  python -m src.analisis_final
"""
import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from src import pipeline as P
from src.datos import RAIZ

K = 2
CARPETA = RAIZ / "data" / "simulacion"


def costo_errores(lotes, aplicar):
    aplicar = np.asarray(aplicar).astype(int)
    y = lotes[P.TARGET].values
    costo_fn, costo_fp = P.costos_por_lote(lotes)
    return ((aplicar == 0) & (y == 1)) * costo_fn.values + ((aplicar == 1) & (y == 0)) * costo_fp.values


def roc(y, p):
    return round(float(roc_auc_score(y, p)), 3) if len(set(y)) == 2 else np.nan


def main():
    sim = pd.read_csv(CARPETA / "lotes_simulacion.csv")
    info = json.loads((RAIZ / "tests" / "fixtures" / "info_modelo.json").read_text())
    linea_base = json.loads((RAIZ / "api" / "linea_base.json").read_text())
    modelo = mlflow.sklearn.load_model(str(RAIZ / "tests" / "fixtures" / "modelo"))

    sim["probabilidad"] = modelo.predict_proba(sim[P.FEATURES])[:, 1]
    rango = sim.groupby("lote_simulacion")["probabilidad"].rank(method="first", ascending=False)
    politicas = {
        "sin_accion": np.zeros(len(sim), dtype=int),
        "regla_estilo": sim["estilo"].isin(linea_base["estilos_alerta"]).astype(int).values,
        "modelo_umbral": (sim["probabilidad"] >= info["umbral_decision"]).astype(int).values,
        "modelo_top2": (rango <= K).astype(int).values,
    }
    for nombre, aplicar in politicas.items():
        sim[f"aplica_{nombre}"] = aplicar
        sim[f"costo_{nombre}"] = costo_errores(sim, aplicar)

    # 1) Lote por lote
    filas = []
    for lote, g in sim.groupby("lote_simulacion"):
        fila = {"lote": lote, "semana": g["semana"].iloc[0], "deriva": int(g["deriva"].max()),
                "lotes": len(g), "merma_alta": int(g[P.TARGET].sum()),
                "roc_auc": roc(g[P.TARGET], g["probabilidad"])}
        for nombre in politicas:
            fila[f"usd_lote_{nombre}"] = round(float(g[f"costo_{nombre}"].mean()), 2)
        fila["clarif_umbral"] = int(g["aplica_modelo_umbral"].sum())
        fila["clarif_top2"] = int(g["aplica_modelo_top2"].sum())
        fila["clarif_regla"] = int(g["aplica_regla_estilo"].sum())
        filas.append(fila)
    por_lote = pd.DataFrame(filas)
    por_lote.to_csv(CARPETA / "por_lote.csv", index=False)

    # 2) Deriva conceptual: contra la CV del PSet 3
    cv = {}
    try:
        P.configurar_mlflow()
        cv = mlflow.MlflowClient().get_run(info["run_id"]).data.metrics
    except Exception as e:   # sin MLflow se imprime igual el resto
        print(f"(No se pudo leer la CV desde MLflow: {e})")
    claves = ["cv_roc_auc_media", "cv_roc_auc_std", "cv_costo_media", "cv_costo_std", "roc_auc_prueba"]
    print("\n=== CV del champion (PSet 3) ===")
    for c in claves:
        print(f"{c}: {cv.get(c, 'no disponible')}")
    if "cv_roc_auc_media" in cv:
        limite = cv["cv_roc_auc_media"] - 2 * cv["cv_roc_auc_std"]
        por_lote["roc_bajo_cv_menos_2sd"] = por_lote["roc_auc"] < limite
        print(f"Límite deriva conceptual (media - 2 desv.): ROC-AUC < {limite:.3f}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== Lote por lote (USD de errores por lote) ===")
    print(por_lote.to_string(index=False))

    # 3) Subgrupos en las semanas normales
    normales = sim[sim["deriva"] == 0]
    filas = []
    for variable in ["cepa", "con_microscopio"]:
        for valor, g in normales.groupby(variable):
            filas.append({
                "subgrupo": f"{variable} = {valor}", "lotes": len(g),
                "tasa_merma_alta": round(float(g[P.TARGET].mean()), 3),
                "roc_auc": roc(g[P.TARGET], g["probabilidad"]),
                "recall_umbral": round(float(recall_score(g[P.TARGET], g["aplica_modelo_umbral"], zero_division=0)), 3),
                "tasa_aplicar_umbral": round(float(g["aplica_modelo_umbral"].mean()), 3),
                "tasa_aplicar_top2": round(float(g["aplica_modelo_top2"].mean()), 3),
                "usd_lote_modelo_umbral": round(float(g["costo_modelo_umbral"].mean()), 2),
                "usd_lote_sin_accion": round(float(g["costo_sin_accion"].mean()), 2),
            })
    subgrupos = pd.DataFrame(filas)
    subgrupos.to_csv(CARPETA / "subgrupos.csv", index=False)
    print("\n=== Subgrupos (semanas normales 1-9) ===")
    print(subgrupos.to_string(index=False))
    print("\nGuardado en data/simulacion/por_lote.csv y data/simulacion/subgrupos.csv")


if __name__ == "__main__":
    main()
    