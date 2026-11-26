"""
Pruebas de comportamiento (secciones 3.3 y 3.4 del PSet 4).

Negocio:
- La levadura nueva (generación 0) no tiene menos riesgo que una levadura reutilizada.
- Un lote en temporada alta no tiene menos riesgo que el mismo lote fuera de ella.
Respaldo:
- Si el modelo no carga, la API responde con la línea base e indica que usó el respaldo.
- Con el interruptor MODO_RESPALDO=1, la API usa la línea base aunque el modelo esté cargado.
"""


# 8. Negocio: en planta, la levadura nueva (generación 0) pierde más que la reutilizada
#    (EDA del PSet 3: mediana 9 % vs 6 %). Reutilizarla no debe SUBIR el riesgo.
def test_levadura_nueva_no_tiene_menos_riesgo(cliente, lote_valido):
    nueva = {**lote_valido, "generacion_levadura": 0, "levadura_nueva": 1}
    reutilizada = {**lote_valido, "generacion_levadura": 4, "levadura_nueva": 0}
    p_nueva = cliente.post("/predecir", json=nueva).json()["probabilidad"]
    p_reutilizada = cliente.post("/predecir", json=reutilizada).json()["probabilidad"]
    assert p_nueva >= p_reutilizada


# 9. Negocio: la temporada alta (abril, mayo, junio y agosto) tiene más lotes con merma alta
#    (EDA: 72 % vs 63 %). El mismo lote en temporada alta no debe tener MENOS riesgo.
def test_temporada_alta_no_baja_el_riesgo(cliente, lote_valido):
    alta = {**lote_valido, "temporada_alta": 1}
    baja = {**lote_valido, "temporada_alta": 0}
    p_alta = cliente.post("/predecir", json=alta).json()["probabilidad"]
    p_baja = cliente.post("/predecir", json=baja).json()["probabilidad"]
    assert p_alta >= p_baja


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


# 7. Interruptor de respaldo (sección 8.4)
def test_interruptor_de_respaldo(cliente, lote_valido, monkeypatch):
    monkeypatch.setenv("MODO_RESPALDO", "1")
    lote_valido["estilo"] = "Iguana"          # Iguana está en los estilos de alerta
    r = cliente.post("/predecir", json=lote_valido)
    cuerpo = r.json()
    assert cuerpo["respaldo"] is True
    assert cuerpo["decision"] == "aplicar_clarificante"
    assert "MODO_RESPALDO" in cuerpo["motivo_respaldo"]
