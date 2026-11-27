"""
Registro de cada predicción y de cada solicitud.

- Siempre en archivos JSONL (registros/predicciones.jsonl y registros/solicitudes.jsonl).
- Si REGISTRAR_SNOWFLAKE=1, también se insertan en Snowflake:
    * cada predicción en PRODUCCION.PREDICCIONES (trazabilidad: versión, run_id, entradas, salida);
    * cada solicitud en PRODUCCION.SOLICITUDES (ruta, código HTTP, latencia y lote de simulación),
      de donde el monitoreo calcula el volumen y la tasa de errores 4xx y 5xx por lote.
- Las inserciones en Snowflake se hacen en un hilo de fondo (una cola): nunca suman latencia
  a la respuesta y, si Snowflake falla, la API sigue funcionando (el JSONL queda como respaldo).
"""
import json
import logging
import os
import queue
import threading
from pathlib import Path

log = logging.getLogger("api.registro")

RUTA_REGISTROS = Path(os.getenv("RUTA_REGISTROS", "registros"))
TABLA = os.getenv("TABLA_PREDICCIONES", "PRODUCCION.PREDICCIONES")
TABLA_SOLICITUDES = os.getenv("TABLA_SOLICITUDES", "PRODUCCION.SOLICITUDES")

_lock = threading.Lock()
_conexion = None
_cola = queue.Queue()
_hilo = None


def _registrar_en_snowflake():
    return os.getenv("REGISTRAR_SNOWFLAKE", "0") == "1"


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
        cur = _conexion.cursor()
        for tabla in (TABLA, TABLA_SOLICITUDES):
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {tabla.split('.')[0]}")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLA} (
                ID_PREDICCION VARCHAR, TIMESTAMP_PREDICCION TIMESTAMP_TZ, NUMERO_DOCUMENTO VARCHAR,
                LOTE_SIMULACION INTEGER, ENTRADAS VARIANT, PROBABILIDAD FLOAT, DECISION VARCHAR,
                VERSION_MODELO VARCHAR, RUN_ID VARCHAR, UMBRAL FLOAT, RESPALDO BOOLEAN,
                MOTIVO_RESPALDO VARCHAR, LATENCIA_MS FLOAT
            )""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLA_SOLICITUDES} (
                TIMESTAMP_SOLICITUD TIMESTAMP_TZ, METODO VARCHAR, RUTA VARCHAR, CODIGO INTEGER,
                LATENCIA_MS FLOAT, LOTE_SIMULACION INTEGER
            )""")
    return _conexion


def _ejecutar(sql, valores):
    """Ejecuta un INSERT. Si falla (por ejemplo, sesión expirada), reconecta y reintenta una vez."""
    global _conexion
    for intento in (1, 2):
        try:
            _conectar().cursor().execute(sql, valores)
            return
        except Exception as e:   # el registro nunca debe tumbar la API: queda en el JSONL
            _conexion = None     # se descarta la conexión; el siguiente intento abre una nueva
            log.warning("No se pudo registrar en Snowflake (intento %d de 2): %s", intento, e)


def _trabajador():
    """Hilo de fondo: toma los INSERT de la cola y los ejecuta en orden, de uno en uno."""
    while True:
        sql, valores = _cola.get()
        try:
            _ejecutar(sql, valores)
        finally:
            _cola.task_done()


def _encolar(sql, valores):
    global _hilo
    with _lock:
        if _hilo is None or not _hilo.is_alive():
            _hilo = threading.Thread(target=_trabajador, name="registro-snowflake", daemon=True)
            _hilo.start()
    _cola.put((sql, valores))


def prediccion(resp, entradas):
    fila = {**resp.model_dump(), "lote_simulacion": entradas.get("lote_simulacion"), "entradas": entradas}
    _escribir_jsonl("predicciones.jsonl", fila)
    if _registrar_en_snowflake():
        _encolar(
            f"INSERT INTO {TABLA} SELECT %s, %s, %s, %s, PARSE_JSON(%s), %s, %s, %s, %s, %s, %s, %s, %s",
            (fila["id_prediccion"], fila["timestamp"], fila["numero_documento"], fila["lote_simulacion"],
             json.dumps(fila["entradas"]), fila["probabilidad"], fila["decision"], fila["version_modelo"],
             fila["run_id"], fila["umbral"], fila["respaldo"], fila["motivo_respaldo"], fila["latencia_ms"]),
        )


def solicitud(fila):
    """fila: timestamp, metodo, ruta, codigo, latencia_ms y lote_simulacion (None si no vino en el encabezado)."""
    _escribir_jsonl("solicitudes.jsonl", fila)
    if _registrar_en_snowflake():
        _encolar(
            f"INSERT INTO {TABLA_SOLICITUDES} SELECT %s, %s, %s, %s, %s, %s",
            (fila["timestamp"], fila["metodo"], fila["ruta"], fila["codigo"], fila["latencia_ms"],
             fila.get("lote_simulacion")),
        )
