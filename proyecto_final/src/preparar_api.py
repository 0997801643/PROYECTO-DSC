"""
Prepara lo que necesita la API a partir del champion del Model Registry.

1. Umbral de decisión para k = 2 lotes clarificados por semana: el percentil de las
   probabilidades del champion en el entrenamiento que deja en "aplicar" a k de los
   lotes que se cocinan en una semana promedio. Se guarda como tag del champion
   (umbral_decision); el umbral del PSet 3 queda en el tag umbral_pset3.
2. Línea base de respaldo (regla de negocio del PSet 1): clarificar los estilos con
   mayor tasa histórica de merma alta, hasta llenar la capacidad de clarificante.
3. Copia del modelo (mlflow.sklearn.save_model) y muestra fija de prueba en
   tests/fixtures/, para correr las pruebas en CI sin servidor de MLflow.

Uso (desde proyecto_final/, con MLflow encendido):  python -m src.preparar_api
"""
import json
import shutil

import mlflow
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

from src import pipeline as P
from src.datos import RAIZ, cargar_obt

NOMBRE_MODELO = "merma-cerveza-modelo"
K_LOTES_SEMANA = 2   # capacidad de clarificante: 2 lotes por semana (supuesto de planta)

RUTA_FIXTURES = RAIZ / "tests" / "fixtures"
RUTA_API = RAIZ / "api"


def main():
    P.configurar_mlflow()
    client = MlflowClient()
    version = client.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    # Se usa el número de versión del champion: get_model_info (MLflow 2.17) no acepta alias
    uri = f"models:/{NOMBRE_MODELO}/{version.version}"
    modelo = mlflow.sklearn.load_model(uri)
    firma = mlflow.models.get_model_info(uri).signature

    df = P.agregar_features(cargar_obt())
    train, test = P.dividir_train_test(df)
    X_tr, y_tr = P.preparar_X_y(train)
    X_te, y_te = P.preparar_X_y(test)

    # 1) Umbral para k lotes por semana
    lotes_semana = train.groupby(train["fecha_coccion"].dt.to_period("W")).size().mean()
    fraccion = min(K_LOTES_SEMANA / lotes_semana, 1.0)
    prob_tr = modelo.predict_proba(X_tr)[:, 1]
    umbral = float(np.quantile(prob_tr, 1 - fraccion))
    # Umbral del PSet 3: se lee del parámetro de la corrida final (no cambia aunque el script se repita)
    umbral_pset3 = client.get_run(version.run_id).data.params.get("umbral_decision", "0.05")

    client.set_model_version_tag(NOMBRE_MODELO, version.version, "umbral_pset3", umbral_pset3)
    client.set_model_version_tag(NOMBRE_MODELO, version.version, "umbral_decision", f"{umbral:.6f}")
    client.set_model_version_tag(NOMBRE_MODELO, version.version, "regla_decision",
                                 f"top-{K_LOTES_SEMANA} lotes por semana (~{fraccion:.0%} de los lotes)")

    # 2) Línea base de respaldo (regla de negocio del PSet 1): clarificar los estilos con
    #    mayor tasa histórica de merma alta, hasta llenar la capacidad de clarificante.
    por_estilo = (train.groupby("estilo")[P.TARGET]
                  .agg(tasa="mean", lotes="size")
                  .sort_values(["tasa", "lotes"], ascending=False))
    por_estilo["participacion"] = por_estilo["lotes"] / len(train)
    estilos_alerta, acumulado = [], 0.0
    for estilo, fila in por_estilo.iterrows():
        if acumulado + fila["participacion"] <= fraccion:
            estilos_alerta.append(estilo)
            acumulado += fila["participacion"]
        else:
            break   # se detiene en el primer estilo que no cabe: no salta a estilos con menos merma
    respaldo = {
        "regla": f"clarificar los estilos con más merma histórica hasta {fraccion:.0%} de los lotes (PSet 1)",
        "estilos_alerta": estilos_alerta,
        "tasa_por_estilo": por_estilo["tasa"].round(4).to_dict(),
        "fraccion_lotes": round(acumulado, 4),
    }
    RUTA_API.mkdir(exist_ok=True)
    (RUTA_API / "linea_base.json").write_text(json.dumps(respaldo, indent=2, ensure_ascii=False))

    # 3) Copia del modelo y muestra fija para CI
    RUTA_FIXTURES.mkdir(parents=True, exist_ok=True)
    ruta_modelo = RUTA_FIXTURES / "modelo"
    if ruta_modelo.exists():
        shutil.rmtree(ruta_modelo)
    mlflow.sklearn.save_model(modelo, str(ruta_modelo), signature=firma)

    muestra = X_te.copy()
    muestra[P.TARGET] = y_te.values
    muestra.to_csv(RUTA_FIXTURES / "muestra_prueba.csv", index=False)

    registradas = client.get_run(version.run_id).data.metrics
    roc_muestra = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])
    info = {
        "nombre": NOMBRE_MODELO,
        "version": version.version,
        "run_id": version.run_id,
        "umbral_decision": umbral,
        "umbral_pset3": float(umbral_pset3),
        "roc_auc_registrado": registradas["roc_auc_prueba"],
        "roc_auc_muestra": roc_muestra,
    }
    (RUTA_FIXTURES / "info_modelo.json").write_text(json.dumps(info, indent=2))

    # Resumen para revisar
    print(f"Champion: versión {version.version} | run_id {version.run_id}")
    print(f"Lotes por semana (entrenamiento): {lotes_semana:.2f} → clarificar el {fraccion:.0%}")
    print(f"Umbral PSet 3: {umbral_pset3}  →  umbral k={K_LOTES_SEMANA}: {umbral:.4f}")
    print(f"Lotes del entrenamiento sobre el umbral: {(prob_tr >= umbral).mean():.0%}")
    print(f"ROC-AUC muestra: {roc_muestra:.6f} | registrado: {registradas['roc_auc_prueba']:.6f}")
    print("\nLínea base por estilo (entrenamiento):")
    print(por_estilo.round(3).to_string())
    print(f"Estilos que clarifica el respaldo: {estilos_alerta} → {acumulado:.0%} de los lotes")


if __name__ == "__main__":
    main()