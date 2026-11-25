"""
Prepara los lotes de la simulación de producción (sección 5 del PSet 4).

- Lotes normales: los lotes del conjunto de prueba (los más recientes, que el modelo no vio),
  agrupados por SEMANA de cocción. Cada semana es un lote de simulación (lote_simulacion = 1, 2, ...).
- Lote con deriva inducida (el último): "temporada alta extrema". Se toman 6 lotes reales y se
  simula que la planta está saturada (+12 lotes cocinados en los 14 días anteriores) y que, por la
  presión, se vuelve a reutilizar la levadura hasta la 4.ª y 5.ª generación (como antes del microscopio).
  Se conserva el numero_documento original para poder unir la merma real.

Salida: data/simulacion/lotes_simulacion.csv (se sube al repo: Kestra lo descarga desde GitHub).

Uso (desde proyecto_final/):  python -m src.preparar_simulacion
"""
import numpy as np
import pandas as pd

from src import pipeline as P
from src.datos import RAIZ, cargar_obt

CARGA_EXTRA = 12           # lotes adicionales en los 14 días anteriores (planta saturada)
GENERACIONES_DERIVA = [4, 5]
N_LOTES_DERIVA = 6

RUTA_SALIDA = RAIZ / "data" / "simulacion" / "lotes_simulacion.csv"
COLUMNAS = (["lote_simulacion", "semana", "deriva", "numero_documento", "fecha_coccion", "estilo"]
            + P.FEATURES + [P.TARGET, "merma_pct", "merma_litros", "cantidad_planificada"])


def main():
    df = P.agregar_features(cargar_obt())
    _, test = P.dividir_train_test(df)
    test = test.copy()

    # Lotes normales: una semana de cocción = un lote de simulación
    test["semana"] = test["fecha_coccion"].dt.to_period("W").dt.start_time
    test["lote_simulacion"] = test["semana"].rank(method="dense").astype(int)
    test["deriva"] = 0

    # Lote con deriva inducida: temporada alta extrema
    rng = np.random.default_rng(P.SEMILLA)
    deriva = test.sample(N_LOTES_DERIVA, random_state=P.SEMILLA).copy()
    deriva["carga_planta_14d"] = deriva["carga_planta_14d"] + CARGA_EXTRA
    deriva["generacion_levadura"] = rng.choice(GENERACIONES_DERIVA, size=len(deriva)).astype(float)
    deriva["levadura_nueva"] = 0.0
    deriva["sin_generacion"] = 0.0
    deriva["lote_simulacion"] = test["lote_simulacion"].max() + 1
    deriva["semana"] = test["semana"].max() + np.timedelta64(7, "D")
    deriva["deriva"] = 1

    sim = pd.concat([test[COLUMNAS].sort_values(["lote_simulacion", "fecha_coccion"]),
                     deriva[COLUMNAS]], ignore_index=True)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(RUTA_SALIDA, index=False)

    # Resumen para revisar
    resumen = sim.groupby("lote_simulacion").agg(
        semana=("semana", "first"), lotes=("numero_documento", "size"), deriva=("deriva", "max"),
        merma_alta=(P.TARGET, "mean"), carga_media=("carga_planta_14d", "mean"),
        generacion_media=("generacion_levadura", "mean"))
    print(resumen.round(2).to_string())
    print(f"\n{len(sim)} filas guardadas en {RUTA_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
    