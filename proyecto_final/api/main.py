"""
API de predicción de merma alta por lote.

- Carga el modelo desde models:/<nombre>@champion. Cada RECARGA_SEGUNDOS revisa si el alias
  cambió y, si cambió, carga la nueva versión: cambiar de versión o hacer rollback no
  requiere reconstruir la imagen.
- El umbral de decisión se lee del tag umbral_decision de la versión en el registro.
- Si el modelo no carga, o si MODO_RESPALDO=1 (interruptor), responde la línea base de
  negocio (estilos con más merma histórica) e indica que usó el respaldo.
- En CI se usa MODELO_RUTA_LOCAL (copia exportada en tests/fixtures/modelo).
"""
import json
import logging
import os
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
    ultima_revision = 0.0


E = Estado()


def modo_respaldo():
    """Interruptor de respaldo: MODO_RESPALDO=1 fuerza la regla de negocio en lugar del modelo."""
    return os.getenv("MODO_RESPALDO", "0").strip().lower() in ("1", "true", "si", "sí")


def _cargar_local():
    import mlflow.sklearn
    info = json.loads((Path(RUTA_LOCAL).parent / "info_modelo.json").read_text())
    E.modelo = mlflow.sklearn.load_model(RUTA_LOCAL)
    E.version, E.run_id, E.umbral = str(info["version"]), info["run_id"], float(info["umbral_decision"])


def _cargar_registro(forzar):
    import mlflow.sklearn
    from mlflow import MlflowClient
    v = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS)
    if forzar or E.modelo is None or str(v.version) != E.version:
        modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{v.version}")
        E.modelo, E.version, E.run_id = modelo, str(v.version), v.run_id
        E.umbral = float(v.tags.get("umbral_decision", "0.5"))
        log.info("Modelo cargado: %s versión %s (@%s), umbral %.4f", NOMBRE, E.version, ALIAS, E.umbral)


def asegurar_modelo(forzar=False):
    """Carga el modelo si falta y revisa el alias cada RECARGA_SEGUNDOS."""
    if not forzar and E.modelo is not None:
        if RUTA_LOCAL or time.time() - E.ultima_revision < RECARGA_SEGUNDOS:
            return
    E.ultima_revision = time.time()
    try:
        _cargar_local() if RUTA_LOCAL else _cargar_registro(forzar)
        E.error = None
    except Exception as e:   # si falla, se mantiene el modelo anterior (o el respaldo si no hay)
        E.error = f"{type(e).__name__}: {e}"
        log.warning("No se pudo cargar el modelo: %s", E.error)


@asynccontextmanager
async def ciclo_de_vida(app):
    asegurar_modelo(forzar=True)
    yield


app = FastAPI(title="API de merma alta por lote", version="1.0", lifespan=ciclo_de_vida)


@app.middleware("http")
async def registrar_solicitud(request: Request, call_next):
    """Registra cada solicitud (ruta, código y latencia) para las métricas de sistema."""
    t0 = time.perf_counter()
    try:
        respuesta = await call_next(request)
        codigo = respuesta.status_code
    except Exception:
        codigo = 500
        raise
    finally:
        registro.solicitud({"timestamp": datetime.now(timezone.utc).isoformat(), "metodo": request.method,
                            "ruta": request.url.path, "codigo": codigo,
                            "latencia_ms": round((time.perf_counter() - t0) * 1000, 2)})
    return respuesta


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
    """Fuerza la revisión del alias (útil para mostrar un rollback sin esperar)."""
    asegurar_modelo(forzar=True)
    return modelo()


@app.post("/predecir", response_model=Prediccion)
def predecir(lote: LoteEntrada, tareas: BackgroundTasks):
    t0 = time.perf_counter()
    asegurar_modelo()
    entradas = lote.model_dump()

    if modo_respaldo() or E.modelo is None:
        motivo = "interruptor MODO_RESPALDO activado" if modo_respaldo() else f"modelo no disponible ({E.error})"
        probabilidad = float(LINEA_BASE["tasa_por_estilo"].get(lote.estilo, 0.0))
        decision = "aplicar_clarificante" if lote.estilo in LINEA_BASE["estilos_alerta"] else "no_aplicar"
        version, run_id, umbral, respaldo = "linea_base", None, None, True
    else:
        X = pd.DataFrame([{f: entradas[f] for f in FEATURES}])
        columnas_num = [f for f in FEATURES if f != "cepa"]
        X[columnas_num] = X[columnas_num].astype(float)   # null → NaN (lo imputa el pipeline)
        probabilidad = float(E.modelo.predict_proba(X)[0, 1])
        decision = "aplicar_clarificante" if probabilidad >= E.umbral else "no_aplicar"
        version, run_id, umbral, respaldo, motivo = E.version, E.run_id, E.umbral, False, None

    resp = Prediccion(
        id_prediccion=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc).isoformat(),
        numero_documento=lote.numero_documento, probabilidad=round(probabilidad, 6),
        decision=decision, accion=ACCIONES[decision], version_modelo=version, run_id=run_id,
        umbral=umbral, respaldo=respaldo, motivo_respaldo=motivo,
        latencia_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
    tareas.add_task(registro.prediccion, resp, entradas)
    return resp