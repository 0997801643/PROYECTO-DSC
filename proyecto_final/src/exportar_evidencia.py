"""
Exporta la evidencia del PSet 4 a docs/ en CSV (sección "Repositorio" del PSet 4).

- docs/predicciones.csv      tabla PRODUCCION.PREDICCIONES de Snowflake
- docs/monitoreo.csv         tabla PRODUCCION.MONITOREO de Snowflake
- docs/alertas.csv           registro de alertas (MONITOREO con ALERTA = TRUE) + severidad,
                             quién la recibe y acción (plantilla de la sección 6.3)
- docs/corridas.csv          todas las corridas del experimento de MLflow (mlflow.search_runs),
                             incluidas las de reentrenamiento
- docs/reentrenamientos.csv  solo las corridas con fase = 'reentrenamiento'

Uso (desde proyecto_final/, con MLflow encendido):  python -m src.exportar_evidencia
"""
import os

import pandas as pd

from src import pipeline as P
from src.datos import RAIZ, conectar

DOCS = RAIZ / "docs"

# Plantilla de alertas (sección 6.3): métrica -> (severidad, quién la recibe, acción)
PLANTILLA = {
    "psi": ("Media", "Analista de datos", "Revisar la fuente; si el desempeño también cae, reentrenar"),
    "pct_fuera_rango": ("Alta", "Analista de datos y jefe de producción",
                        "No usar el modelo esa semana; decidir con la regla por estilo"),
    "pct_nulos": ("Media", "Responsable de la bitácora", "Completar el registro en planta"),
    "pct_categoria_nueva": ("Alta", "Analista de datos", "Revisar la cepa nueva; usar la regla por estilo"),
    "psi_probabilidad": ("Media", "Analista de datos", "Revisar junto con la deriva de las features"),
    "costo_por_lote_modelo": ("Alta", "Analista de datos", "Reentrenar si se repite 2 lotes seguidos"),
    "roc_auc": ("Alta", "Analista de datos", "Reentrenar; si persiste, modo respaldo"),
    "clarificantes_semana": ("Media", "Jefe de producción", "Aplicar solo a los 2 lotes de mayor probabilidad"),
    "latencia_p95_ms": ("Baja", "Soporte de la API", "Revisar arranque en frío y recursos"),
    "pct_respaldo": ("Alta", "Soporte de la API", "Revisar por qué el modelo no cargó"),
    "pct_predicciones_faltantes": ("Alta", "Soporte de la API", "Reenviar el lote y revisar Snowflake"),
}


def consulta(con, sql):
    cur = con.cursor()
    cur.execute(sql)
    return pd.DataFrame(cur.fetchall(), columns=[c[0].lower() for c in cur.description])


def main():
    DOCS.mkdir(exist_ok=True)

    # 1) Snowflake
    with conectar() as con:
        pred = consulta(con, "SELECT * FROM PRODUCCION.PREDICCIONES ORDER BY TIMESTAMP_PREDICCION")
        mon = consulta(con, "SELECT * FROM PRODUCCION.MONITOREO ORDER BY LOTE_SIMULACION, TIPO, METRICA")
    pred.to_csv(DOCS / "predicciones.csv", index=False)
    mon.to_csv(DOCS / "monitoreo.csv", index=False)

    alertas = mon[mon["alerta"].astype(bool)].copy()
    alertas["severidad"] = alertas["metrica"].map(lambda m: PLANTILLA.get(m, ("", "", ""))[0])
    alertas["quien_la_recibe"] = alertas["metrica"].map(lambda m: PLANTILLA.get(m, ("", "", ""))[1])
    alertas["accion"] = alertas["metrica"].map(lambda m: PLANTILLA.get(m, ("", "", ""))[2])
    alertas.to_csv(DOCS / "alertas.csv", index=False)

    # 2) MLflow: todas las corridas del experimento, incluidas las de reentrenamiento
    mlflow = P.configurar_mlflow()
    corridas = mlflow.search_runs(experiment_names=[os.getenv("MLFLOW_EXPERIMENT", "merma-cerveza")])
    corridas.to_csv(DOCS / "corridas.csv", index=False)
    fase = corridas.get("tags.fase", pd.Series(index=corridas.index, dtype=object))
    reentrenamientos = corridas[fase == "reentrenamiento"]
    reentrenamientos.to_csv(DOCS / "reentrenamientos.csv", index=False)

    print(f"predicciones.csv      {len(pred)} filas")
    print(f"monitoreo.csv         {len(mon)} filas")
    print(f"alertas.csv           {len(alertas)} alertas")
    print(f"corridas.csv          {len(corridas)} corridas")
    print(f"reentrenamientos.csv  {len(reentrenamientos)} corridas de reentrenamiento")


if __name__ == "__main__":
    main()
