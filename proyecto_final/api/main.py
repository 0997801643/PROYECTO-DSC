"""
API de predicción de merma alta por lote.

- Carga el modelo desde models:/<nombre>@champion. Un hilo de fondo revisa el alias cada
  RECARGA_SEGUNDOS y, si cambió, carga la nueva versión: cambiar de versión o hacer rollback no
  requiere reconstruir la imagen. La revisión NO ocurre dentro de una predicción, así que no
  suma latencia a ninguna solicitud. POST /recargar fuerza la revisión en el momento.
- El umbral de decisión se lee del tag umbral_decision de la versión en el registro.
- Si el modelo no carga, o si MODO_RESPALDO=1 (interruptor), responde la línea base de
  negocio (estilos con más merma histórica) e indica que usó el respaldo.
- Cada solicitud (ruta, código HTTP, latencia y lote de simulación, que llega en el encabezado
  X-Lote-Simulacion) se registra para las métricas de sistema: volumen y errores 4xx y 5xx.
- En CI se usa MODELO_RUTA_LOCAL (copia exportada en tests/fixtures/modelo).
"""
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import registro
from api.esquemas import FEATURES, LoteEntrada, Prediccion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

NOMBRE = os.getenv("MODELO_NOMBRE", "merma-cerveza-modelo")
ALIAS = os.getenv("MODELO_ALIAS", "champion")
RUTA_LOCAL = os.getenv("MODELO_RUTA_LOCAL")            # solo en CI / pruebas
RECARGA_SEGUNDOS = float(os.getenv("RECARGA_SEGUNDOS", "30"))
LINEA_BASE = json.loads((Path(__file__).parent / "linea_base.json").read_text(encoding="utf-8"))

ACCIONES = {
    "aplicar_clarificante": "Aplicar clarificante al lote antes del filtrado. "
                            "El jefe de producción confirma con lo que observa en el tanque.",
    "no_aplicar": "Clarificación estándar, sin clarificante adicional.",
}


class Estado:
    modelo = None
    version = None
    run_id = None
    umbral = None
    error = None


E = Estado()
_lock_modelo = threading.Lock()     # el modelo, su versión y su umbral cambian juntos
_detener = threading.Event()


def modo_respaldo():
    """Interruptor de respaldo: MODO_RESPALDO=1 fuerza la regla de negocio en lugar del modelo."""
    return os.getenv("MODO_RESPALDO", "0").strip().lower() in ("1", "true", "si", "sí")


def _cargar_local():
    import mlflow.sklearn
    info = json.loads((Path(RUTA_LOCAL).parent / "info_modelo.json").read_text())
    modelo = mlflow.sklearn.load_model(RUTA_LOCAL)
    with _lock_modelo:
        E.modelo = modelo
        E.version, E.run_id, E.umbral = str(info["version"]), info["run_id"], float(info["umbral_decision"])


def _cargar_registro(forzar):
    import mlflow.sklearn
    from mlflow import MlflowClient
    v = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS)
    if forzar or E.modelo is None or str(v.version) != E.version:
        modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{v.version}")
        with _lock_modelo:
            E.modelo, E.version, E.run_id = modelo, str(v.version), v.run_id
            E.umbral = float(v.tags.get("umbral_decision", "0.5"))
        log.info("Modelo cargado: %s versión %s (@%s), umbral %.4f", NOMBRE, E.version, ALIAS, E.umbral)


def cargar_modelo(forzar=False):
    """Carga el modelo (copia local en CI o Model Registry). Si falla, se mantiene el modelo
    anterior; si no había ninguno, las predicciones usan el respaldo."""
    try:
        _cargar_local() if RUTA_LOCAL else _cargar_registro(forzar)
        E.error = None
    except Exception as e:
        E.error = f"{type(e).__name__}: {e}"
        log.warning("No se pudo cargar el modelo: %s", E.error)


def asegurar_modelo():
    """Solo intenta cargar si todavía no hay modelo. Con modelo cargado no hace nada (no consulta MLflow)."""
    if E.modelo is None:
        cargar_modelo(forzar=True)


def _revisar_alias_periodicamente():
    """Hilo de fondo: cada RECARGA_SEGUNDOS revisa si el alias apunta a otra versión."""
    while not _detener.wait(RECARGA_SEGUNDOS):
        cargar_modelo(forzar=False)


@asynccontextmanager
async def ciclo_de_vida(app):
    cargar_modelo(forzar=True)
    hilo = None
    if not RUTA_LOCAL:   # en CI el modelo es una copia fija: no hay alias que revisar
        _detener.clear()
        hilo = threading.Thread(target=_revisar_alias_periodicamente, name="revisar-alias", daemon=True)
        hilo.start()
    yield
    _detener.set()


app = FastAPI(title="API de merma alta por lote", version="1.1", lifespan=ciclo_de_vida)


def _lote_del_encabezado(request):
    """Lote de simulación que envía produccion_lotes en el encabezado X-Lote-Simulacion (o None)."""
    valor = request.headers.get("x-lote-simulacion")
    try:
        return int(valor) if valor not in (None, "") else None
    except ValueError:
        return None


@app.middleware("http")
async def registrar_solicitud(request: Request, call_next):
    """Registra cada solicitud (ruta, código, latencia y lote) para las métricas de sistema."""
    t0 = time.perf_counter()
    codigo = 500                       # si call_next lanza una excepción, cuenta como error 5xx
    try:
        respuesta = await call_next(request)
        codigo = respuesta.status_code
        return respuesta
    finally:
        registro.solicitud({"timestamp": datetime.now(timezone.utc).isoformat(), "metodo": request.method,
                            "ruta": request.url.path, "codigo": codigo,
                            "latencia_ms": round((time.perf_counter() - t0) * 1000, 2),
                            "lote_simulacion": _lote_del_encabezado(request)})


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError):
    """422 con un mensaje que dice qué campo falló."""
    errores = [{"campo": ".".join(str(p) for p in e["loc"] if p != "body"), "mensaje": e["msg"]}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Entrada inválida", "detalle": errores})


@app.get("/salud")
def salud():
    asegurar_modelo()
    return {"estado": "ok" if E.modelo is not None else "degradado",
            "modelo_cargado": E.modelo is not None,
            "modo_respaldo": modo_respaldo(),
            "error_carga": E.error}


@app.get("/modelo")
def modelo():
    asegurar_modelo()
    return {"nombre": NOMBRE, "alias": ALIAS, "version": E.version, "run_id": E.run_id,
            "umbral_decision": E.umbral, "fuente": "copia local" if RUTA_LOCAL else "Model Registry",
            "modelo_cargado": E.modelo is not None}


@app.post("/recargar")
def recargar():
    """Fuerza la revisión del alias (útil para mostrar un rollback sin esperar al hilo de fondo)."""
    cargar_modelo(forzar=True)
    return modelo()


@app.post("/predecir", response_model=Prediccion)
def predecir(lote: LoteEntrada, tareas: BackgroundTasks):
    t0 = time.perf_counter()
    asegurar_modelo()
    entradas = lote.model_dump()
    with _lock_modelo:   # el mismo modelo, versión y umbral durante toda la predicción
        modelo_actual, version, run_id, umbral = E.modelo, E.version, E.run_id, E.umbral

    if modo_respaldo() or modelo_actual is None:
        motivo = "interruptor MODO_RESPALDO activado" if modo_respaldo() else f"modelo no disponible ({E.error})"
        probabilidad = float(LINEA_BASE["tasa_por_estilo"].get(lote.estilo, 0.0))
        decision = "aplicar_clarificante" if lote.estilo in LINEA_BASE["estilos_alerta"] else "no_aplicar"
        version, run_id, umbral, respaldo = "linea_base", None, None, True
    else:
        X = pd.DataFrame([{f: entradas[f] for f in FEATURES}])
        columnas_num = [f for f in FEATURES if f != "cepa"]
        X[columnas_num] = X[columnas_num].astype(float)   # null → NaN (lo imputa el pipeline)
        probabilidad = float(modelo_actual.predict_proba(X)[0, 1])
        decision = "aplicar_clarificante" if probabilidad >= umbral else "no_aplicar"
        respaldo, motivo = False, None

    resp = Prediccion(
        id_prediccion=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc).isoformat(),
        numero_documento=lote.numero_documento, probabilidad=round(probabilidad, 6),
        decision=decision, accion=ACCIONES[decision], version_modelo=version, run_id=run_id,
        umbral=umbral, respaldo=respaldo, motivo_respaldo=motivo,
        latencia_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
    tareas.add_task(registro.prediccion, resp, entradas)
    return resp
