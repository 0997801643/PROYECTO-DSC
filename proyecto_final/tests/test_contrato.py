"""
Pruebas de contrato de la API (sección 3.1 del PSet 4).

1. Entrada válida → 200 con la estructura esperada y una decisión coherente con el umbral.
2. Falta un campo → 422 que dice qué campo falló.
3. Tipo incorrecto → 422.
4. Categoría desconocida → 422.
"""
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
