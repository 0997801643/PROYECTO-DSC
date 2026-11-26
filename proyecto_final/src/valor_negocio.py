"""
Valor de negocio (sección 9 del PSet 4).

Compara, en los lotes de la simulación de producción (con su merma real), cuánto cuestan los
errores de cada política para decidir el clarificante:

1. Sin acción (hoy): nunca se aplica clarificante.
2. Regla por estilo (línea base del PSet 1 y respaldo de la API).
3. Modelo con umbral fijo (lo que hace hoy la API: probabilidad >= umbral_decision).
4. Modelo top-2 por semana: cada semana se clarifican solo los 2 lotes con mayor probabilidad
   (respeta siempre el presupuesto de k = 2).

Costo de cada lote (igual que el PSet 3 y el monitoreo):
- merma alta SIN clarificante  -> litros sobre la meta del 5 % x precio por litro del estilo
- lote normal CON clarificante -> costo del clarificante desperdiciado

Usa el champion guardado en tests/fixtures/modelo (versión 1), así no necesita MLflow.

Salida: tabla en pantalla y data/simulacion/valor_negocio.csv
Uso (desde proyecto_final/):  python -m src.valor_negocio
"""
import json

import mlflow
import numpy as np
import pandas as pd

from src import pipeline as P
from src.datos import RAIZ

K = 2
RUTA_SIM = RAIZ / "data" / "simulacion" / "lotes_simulacion.csv"
RUTA_SALIDA = RAIZ / "data" / "simulacion" / "valor_negocio.csv"


def resumen(nombre, lotes, aplicar):
    aplicar = np.asarray(aplicar).astype(int)
    y = lotes[P.TARGET].values
    costo_fn, costo_fp = P.costos_por_lote(lotes)
    costo = ((aplicar == 0) & (y == 1)) * costo_fn.values + ((aplicar == 1) & (y == 0)) * costo_fp.values
    por_semana = pd.Series(aplicar).groupby(lotes["lote_simulacion"].values).sum()
    return {
        "politica": nombre,
        "costo_total_usd": round(float(costo.sum()), 2),
        "costo_por_lote_usd": round(float(costo.mean()), 2),
        "clarificantes_usados": int(aplicar.sum()),
        "semanas_sobre_presupuesto": int((por_semana > K).sum()),
        "merma_alta_detectada": f"{int(((aplicar == 1) & (y == 1)).sum())} de {int(y.sum())}",
        "clarificante_innecesario": int(((aplicar == 1) & (y == 0)).sum()),
    }


def main():
    sim = pd.read_csv(RUTA_SIM)
    info = json.loads((RAIZ / "tests" / "fixtures" / "info_modelo.json").read_text())
    linea_base = json.loads((RAIZ / "api" / "linea_base.json").read_text())
    modelo = mlflow.sklearn.load_model(str(RAIZ / "tests" / "fixtures" / "modelo"))

    sim["probabilidad"] = modelo.predict_proba(sim[P.FEATURES])[:, 1]
    # Top-2 por semana: los 2 lotes con mayor probabilidad de cada lote de simulación
    sim["rango_semana"] = sim.groupby("lote_simulacion")["probabilidad"].rank(method="first", ascending=False)

    filas = []
    for nombre_periodo, lotes in [("semanas normales (1-9)", sim[sim["deriva"] == 0]),
                                  ("semana con deriva (10)", sim[sim["deriva"] == 1]),
                                  ("todas (1-10)", sim)]:
        lotes = lotes.reset_index(drop=True)
        politicas = [
            ("sin acción (hoy)", np.zeros(len(lotes))),
            ("regla por estilo (PSet 1)", lotes["estilo"].isin(linea_base["estilos_alerta"])),
            ("modelo, umbral fijo", lotes["probabilidad"] >= info["umbral_decision"]),
            (f"modelo, top-{K} por semana", lotes["rango_semana"] <= K),
        ]
        for nombre, aplicar in politicas:
            filas.append({"periodo": f"{nombre_periodo}, {len(lotes)} lotes", **resumen(nombre, lotes, aplicar)})

    tabla = pd.DataFrame(filas)
    base = tabla[tabla["politica"] == "sin acción (hoy)"].set_index("periodo")["costo_total_usd"]
    tabla["ahorro_vs_hoy_usd"] = (tabla["periodo"].map(base) - tabla["costo_total_usd"]).round(2)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_SALIDA, index=False)

    pd.set_option("display.width", 200)
    for periodo, grupo in tabla.groupby("periodo", sort=False):
        print(f"\n=== {periodo} ===")
        print(grupo.drop(columns="periodo").to_string(index=False))
    print(f"\nGuardado en {RUTA_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()