"""
Esquemas de entrada y salida de la API.

La entrada coincide con la firma del modelo registrado en MLflow (mismas 10 features,
mismos tipos). Los rangos son los valores POSIBLES en la planta: un valor imposible
(mes 13, generación negativa, cepa que no existe) se rechaza con 422. Un valor posible
pero fuera de lo visto en el entrenamiento se acepta y lo detecta el monitoreo.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Mismas features y mismo orden que src/pipeline.py (FEATURES)
FEATURES = [
    "generacion_levadura", "n_cocciones", "coccion_mes", "coccion_dia_semana", "carga_planta_14d",
    "levadura_nueva", "sin_generacion", "con_microscopio", "temporada_alta",
    "cepa",
]

Cepa = Literal["BE-256", "K-97", "US-05", "W34/70"]
Estilo = Literal["American Lager", "Oktoberfest", "Lagarto", "Red Llama", "Condor",
                 "Tortuga", "Iguana", "Belgian Tripel", "Honey Bock"]


class LoteEntrada(BaseModel):
    """Un lote de cerveza el día de su cocción."""
    model_config = ConfigDict(
        extra="forbid",   # un campo que no existe en el esquema también es un error
        json_schema_extra={"example": {
            "numero_documento": "1301", "estilo": "American Lager",
            "generacion_levadura": 2, "n_cocciones": 2, "coccion_mes": 9, "coccion_dia_semana": 3,
            "carga_planta_14d": 8, "levadura_nueva": 0, "sin_generacion": 0, "con_microscopio": 1,
            "temporada_alta": 1, "cepa": "W34/70",
        }},
    )

    # Identificación del lote (no son features del modelo)
    numero_documento: str = Field(..., min_length=1, description="Orden de producción en SAP")
    estilo: Estilo = Field(..., description="Estilo de cerveza (lo usa la línea base de respaldo)")
    lote_simulacion: Optional[int] = Field(None, ge=0, description="Período de la simulación (opcional)")

    # Features del modelo (firma registrada en MLflow)
    generacion_levadura: Optional[float] = Field(..., ge=0, le=10, description="Generación de la levadura; null si no se registró")
    n_cocciones: float = Field(..., gt=0, le=10, description="Cocciones del lote")
    coccion_mes: int = Field(..., ge=1, le=12, description="Mes de cocción")
    coccion_dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana de la cocción")
    carga_planta_14d: int = Field(..., ge=0, le=100, description="Lotes cocinados en los 14 días anteriores")
    levadura_nueva: Optional[int] = Field(..., ge=0, le=1, description="1 si la levadura es generación 0; null si no se sabe")
    sin_generacion: int = Field(..., ge=0, le=1, description="1 si no se registró la generación")
    con_microscopio: int = Field(..., ge=0, le=1, description="1 si el lote tiene medición de microscopio")
    temporada_alta: int = Field(..., ge=0, le=1, description="1 si el mes es de temporada alta")
    cepa: Cepa = Field(..., description="Cepa de levadura")


class Prediccion(BaseModel):
    id_prediccion: str
    timestamp: str
    numero_documento: str
    probabilidad: float = Field(..., description="Probabilidad de merma alta (> 5 %)")
    decision: Literal["aplicar_clarificante", "no_aplicar"]
    accion: str = Field(..., description="Qué hacer con el lote (acción del PSet 1)")
    version_modelo: str
    run_id: Optional[str]
    umbral: Optional[float]
    respaldo: bool = Field(..., description="True si respondió la línea base y no el modelo")
    motivo_respaldo: Optional[str]
    latencia_ms: float
    