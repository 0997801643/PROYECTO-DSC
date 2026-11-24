"""
Configuración común de las pruebas.

Las pruebas corren SIN servidor de MLflow y SIN Snowflake (como en GitHub Actions):
- la API usa la copia exportada del modelo en tests/fixtures/modelo (MODELO_RUTA_LOCAL);
- los registros van a una carpeta temporal y no se envían a Snowflake.
Estas variables se fijan ANTES de importar la API.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]          # proyecto_final/
FIXTURES = RAIZ / "tests" / "fixtures"

sys.path.insert(0, str(RAIZ))                      # para poder hacer "import api"
os.environ["MODELO_RUTA_LOCAL"] = str(FIXTURES / "modelo")
os.environ["RUTA_REGISTROS"] = tempfile.mkdtemp(prefix="registros_prueba_")
os.environ["REGISTRAR_SNOWFLAKE"] = "0"
os.environ["MODO_RESPALDO"] = "0"


@pytest.fixture
def cliente():
    """Cliente de prueba de la API (con el modelo de tests/fixtures cargado al arrancar)."""
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def lote_valido():
    """Un lote correcto, igual al ejemplo de la documentación de la API."""
    return {
        "numero_documento": "1301", "estilo": "American Lager",
        "generacion_levadura": 2, "n_cocciones": 2, "coccion_mes": 9, "coccion_dia_semana": 3,
        "carga_planta_14d": 8, "levadura_nueva": 0, "sin_generacion": 0, "con_microscopio": 1,
        "temporada_alta": 1, "cepa": "W34/70",
    }