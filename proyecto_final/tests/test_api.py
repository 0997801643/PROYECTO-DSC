"""
Pruebas automáticas de la API (sección 4.1 del PSet 4).

1. Entrada válida → 200 con la estructura esperada.
2. Falta un campo → 422 que dice qué campo falló.
3. Tipo incorrecto → 422.
4. Categoría desconocida → 422.
5. El modelo reproduce la métrica registrada en MLflow sobre la muestra fija.
6. Si el modelo no carga, la API responde con la línea base (respaldo).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from conftest import FIXTURES

CAMPOS_RESPUESTA = {
    "id_prediccion", "timestamp", "numero_documento", "probabilidad", "decision", "accion",
    "version_modelo", "run_id", "umbral", "respaldo", "motivo_respaldo", "latencia_ms",
}


def campos_con_error(respuesta):
    return [e["campo"] for e in respuesta.json()["detalle"]]


# 1. Entrada válida
def test_entrada_valida_devuelve_prediccion(cliente, lote_valido):
    r = cliente.post("/predecir", json=lote_valido)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == CAMPOS_RESPUESTA
    assert 0.0 <= cuerpo["probabilidad"] <= 1.0
    assert cuerpo["decision"] in ("aplicar_clarificante", "no_aplicar")
    assert cuerpo["respaldo"] is False
    # La decisión es coherente con el umbral
    esperada = "aplicar_clarificante" if cuerpo["probabilidad"] >= cuerpo["umbral"] else "no_aplicar"
    assert cuerpo["decision"] == esperada


# 2. Falta un campo
def test_falta_un_campo_devuelve_422(cliente, lote_valido):
    del lote_valido["cepa"]
    r = cliente.post("/predecir", json=lote_valido)
    assert r.status_code == 422
    assert "cepa" in campos_con_error(r)


# 3. Tipo incorrecto
def test_tipo_incorrecto_devuelve_422(cliente, lote_valido):
    lote_valido["coccion_mes"] = "septiembre"
    r = cliente.post("/predecir", json=lote_valido)
    assert r.status_code == 422
    assert "coccion_mes" in campos_con_error(r)


# 4. Categoría desconocida
def test_categoria_desconocida_devuelve_422(cliente, lote_valido):
    lote_valido["cepa"] = "XYZ"
    r = cliente.post("/predecir", json=lote_valido)
    assert r.status_code == 422
    assert "cepa" in campos_con_error(r)


# 5. El modelo reproduce la métrica registrada
def test_modelo_reproduce_metrica_registrada():
    import mlflow.sklearn
    info = json.loads((FIXTURES / "info_modelo.json").read_text())
    modelo = mlflow.sklearn.load_model(str(FIXTURES / "modelo"))
    muestra = pd.read_csv(FIXTURES / "muestra_prueba.csv")
    X, y = muestra.drop(columns="merma_alta"), muestra["merma_alta"]
    roc = roc_auc_score(y, modelo.predict_proba(X)[:, 1])
    assert np.isclose(roc, info["roc_auc_registrado"], atol=1e-6), (
        f"ROC-AUC {roc:.6f} no coincide con el registrado {info['roc_auc_registrado']:.6f}")


# 6. Respaldo si el modelo no carga
def test_respaldo_si_el_modelo_no_carga(cliente, lote_valido, monkeypatch):
    from api import main
    monkeypatch.setattr(main, "RUTA_LOCAL", "/ruta/que/no/existe")
    monkeypatch.setattr(main.E, "modelo", None)
    r = cliente.post("/predecir", json=lote_valido)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["respaldo"] is True
    assert cuerpo["version_modelo"] == "linea_base"
    assert "modelo no disponible" in cuerpo["motivo_respaldo"]


# Extra: el interruptor de respaldo (sección 8.4)
def test_interruptor_de_respaldo(cliente, lote_valido, monkeypatch):
    monkeypatch.setenv("MODO_RESPALDO", "1")
    lote_valido["estilo"] = "Iguana"          # Iguana está en los estilos de alerta
    r = cliente.post("/predecir", json=lote_valido)
    cuerpo = r.json()
    assert cuerpo["respaldo"] is True
    assert cuerpo["decision"] == "aplicar_clarificante"
    assert "MODO_RESPALDO" in cuerpo["motivo_respaldo"]
    