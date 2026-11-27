"""
Carga los lotes de la simulación en Snowflake (sección 5 del PSet 4).

Sube data/simulacion/lotes_simulacion.csv (lo genera src/preparar_simulacion.py) a la tabla
PRODUCCION.LOTES_SIMULACION. El flujo de Kestra produccion_lotes lee cada lote desde esa tabla.

- La tabla se reemplaza completa en cada carga (los lotes de la simulación son fijos).
- Guarda también la merma real, pero produccion_lotes solo lee las columnas de entrada del
  modelo: la etiqueta la une el monitoreo con RETRASO_ETIQUETAS semanas de retraso.

Uso (desde proyecto_final/):  python -m src.cargar_lotes
"""
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from src.datos import RAIZ, conectar

RUTA_LOTES = RAIZ / "data" / "simulacion" / "lotes_simulacion.csv"
ESQUEMA = "PRODUCCION"
TABLA = "LOTES_SIMULACION"


def main():
    lotes = pd.read_csv(RUTA_LOTES, dtype={"numero_documento": str, "semana": str, "fecha_coccion": str})
    lotes.columns = [c.upper() for c in lotes.columns]   # Snowflake guarda los nombres en mayúsculas

    with conectar() as con:
        con.cursor().execute(f"CREATE SCHEMA IF NOT EXISTS {ESQUEMA}")
        ok, _, filas, _ = write_pandas(con, lotes, table_name=TABLA, schema=ESQUEMA,
                                       auto_create_table=True, overwrite=True, quote_identifiers=False)
        cur = con.cursor()
        cur.execute(f"""SELECT LOTE_SIMULACION, COUNT(*), MAX(DERIVA)
                        FROM {ESQUEMA}.{TABLA} GROUP BY 1 ORDER BY 1""")
        resumen = cur.fetchall()

    if not ok:
        raise SystemExit("No se pudo cargar la tabla en Snowflake")
    print(f"{filas} filas cargadas en {ESQUEMA}.{TABLA}")
    for lote, n, deriva in resumen:
        print(f"  lote {lote:>2}: {n} lotes de cerveza" + ("  <- deriva inducida" if deriva else ""))


if __name__ == "__main__":
    main()
