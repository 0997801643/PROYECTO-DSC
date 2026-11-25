"""
Registro de cada predicción y de cada solicitud.

- Siempre en archivos JSONL (registros/predicciones.jsonl y registros/solicitudes.jsonl).
- Si REGISTRAR_SNOWFLAKE=1, cada predicción también se inserta en la tabla PREDICCIONES
  de Snowflake. Se hace en segundo plano: no suma latencia a la respuesta.
"""
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("api.registro")

RUTA_REGISTROS = Path(os.getenv("RUTA_REGISTROS", "registros"))
TABLA = os.getenv("TABLA_PREDICCIONES", "PRODUCCION.PREDICCIONES")

_lock = threading.Lock()
_conexion = None


def _escribir_jsonl(nombre, fila):
    RUTA_REGISTROS.mkdir(parents=True, exist_ok=True)
    with _lock, open(RUTA_REGISTROS / nombre, "a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def _llave_privada():
    from cryptography.hazmat.primitives import serialization
    ruta = Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])
    llave = serialization.load_pem_private_key(ruta.read_bytes(), password=None)
    return llave.private_bytes(encoding=serialization.Encoding.DER,
                               format=serialization.PrivateFormat.PKCS8,
                               encryption_algorithm=serialization.NoEncryption())


def _conectar():
    global _conexion
    if _conexion is None:
        import snowflake.connector
        _conexion = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
            role=os.environ["SNOWFLAKE_ROLE"], warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"], private_key=_llave_privada(),
            client_session_keep_alive=True,   # evita que la sesión expire si la API pasa horas sin predecir
        )
        esquema = TABLA.split(".")[0]
        cur = _conexion.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {esquema}")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLA} (
                ID_PREDICCION VARCHAR, TIMESTAMP_PREDICCION TIMESTAMP_TZ, NUMERO_DOCUMENTO VARCHAR,
                LOTE_SIMULACION INTEGER, ENTRADAS VARIANT, PROBABILIDAD FLOAT, DECISION VARCHAR,
                VERSION_MODELO VARCHAR, RUN_ID VARCHAR, UMBRAL FLOAT, RESPALDO BOOLEAN,
                MOTIVO_RESPALDO VARCHAR, LATENCIA_MS FLOAT
            )""")
    return _conexion


def _insertar_snowflake(fila):
    """Inserta la predicción. Si falla (por ejemplo, sesión expirada), reconecta y reintenta una vez."""
    global _conexion
    sql = f"""INSERT INTO {TABLA} SELECT %s, %s, %s, %s, PARSE_JSON(%s), %s, %s, %s, %s, %s, %s, %s, %s"""
    valores = (fila["id_prediccion"], fila["timestamp"], fila["numero_documento"], fila["lote_simulacion"],
               json.dumps(fila["entradas"]), fila["probabilidad"], fila["decision"], fila["version_modelo"],
               fila["run_id"], fila["umbral"], fila["respaldo"], fila["motivo_respaldo"], fila["latencia_ms"])
    for intento in (1, 2):
        try:
            with _lock:
                _conectar().cursor().execute(sql, valores)
            return
        except Exception as e:   # el registro nunca debe tumbar la API: queda en el JSONL
            _conexion = None     # se descarta la conexión; el siguiente intento abre una nueva
            log.warning("No se pudo registrar en Snowflake (intento %d de 2): %s", intento, e)


def prediccion(resp, entradas):
    fila = {**resp.model_dump(), "lote_simulacion": entradas.get("lote_simulacion"), "entradas": entradas}
    _escribir_jsonl("predicciones.jsonl", fila)
    if os.getenv("REGISTRAR_SNOWFLAKE", "0") == "1":
        _insertar_snowflake(fila)


def solicitud(fila):
    _escribir_jsonl("solicitudes.jsonl", fila)