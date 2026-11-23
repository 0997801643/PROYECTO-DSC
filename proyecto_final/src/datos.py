"""
Carga de la OBT desde Snowflake y metadatos de los datos para MLflow.

La primera vez se consulta Snowflake y se guarda un snapshot en
proyecto_final/data/obt_snapshot.parquet. Las siguientes veces se lee el snapshot,
así todos los notebooks usan exactamente los mismos datos (mismo hash).
Para volver a consultar Snowflake: cargar_obt(forzar=True).
"""
import hashlib
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]   # proyecto_final/
load_dotenv(RAIZ / ".env")                    # Snowflake + MLflow

TABLA_OBT = "CERVECERIA_DB.OBT.OBT_MERMA_LOTES"
RUTA_SNAPSHOT = RAIZ / "data" / "obt_snapshot.parquet"

# Definición del target (negocio): merma > 5 % indica falla del proceso
UMBRAL_MERMA_ALTA = 5.0
TARGET = "merma_alta"


def _llave_privada():
    """Lee la llave privada del PSet 2 y la convierte al formato que pide el conector."""
    from cryptography.hazmat.primitives import serialization

    ruta = Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    llave = serialization.load_pem_private_key(ruta.read_bytes(), password=None)
    return llave.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def conectar():
    import snowflake.connector

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        private_key=_llave_privada(),
    )


def _leer_snowflake():
    with conectar() as con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {TABLA_OBT}")
        df = cur.fetch_pandas_all()
    df.columns = df.columns.str.lower()
    for col in [c for c in df.columns if c.startswith("fecha")]:
        df[col] = pd.to_datetime(df[col])
    return df


def cargar_obt(forzar=False):
    """Devuelve la OBT ordenada por fecha de cocción, con la columna target 'merma_alta' (0/1)."""
    if forzar or not RUTA_SNAPSHOT.exists():
        df = _leer_snowflake()
        RUTA_SNAPSHOT.parent.mkdir(exist_ok=True)
        df.to_parquet(RUTA_SNAPSHOT, index=False)
    df = pd.read_parquet(RUTA_SNAPSHOT)
    df = df.sort_values(["fecha_coccion", "numero_documento"]).reset_index(drop=True)
    df[TARGET] = (df["merma_pct"] > UMBRAL_MERMA_ALTA).astype(int)
    return df


def hash_dataset(df):
    """Huella del dataset: si cambia un solo valor, cambia el hash."""
    valores = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(valores.tobytes()).hexdigest()[:16]


def metadatos_datos(df):
    """Lo que cada corrida de MLflow registra sobre los datos."""
    return {
        "tabla_origen": TABLA_OBT,
        "fecha_corte": str(df["fecha_cierre"].max().date()),
        "hash_dataset": hash_dataset(df),
        "n_filas": len(df),
    }