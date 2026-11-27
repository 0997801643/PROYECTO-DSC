# merma-cerveza

[![CI](https://github.com/0997801643/PROYECTO-DSC/actions/workflows/ci.yml/badge.svg)](https://github.com/0997801643/PROYECTO-DSC/actions/workflows/ci.yml)
[![CD](https://github.com/0997801643/PROYECTO-DSC/actions/workflows/cd.yml/badge.svg)](https://github.com/0997801643/PROYECTO-DSC/actions/workflows/cd.yml)

Predice, el día de la cocción, si un lote de cerveza tendrá **merma de filtración > 5 %**, para decidir a qué lotes aplicar clarificante con un presupuesto de **2 lotes por semana**.

Proyecto final del curso *Fundamentos de Ciencia de Datos: End to End* (PSets 1 a 4).

## Arquitectura

```
SAP (CSV) ─► Kestra ─► RAW ─► dbt (Bronze ─► Silver ─► Gold) ─► Spark (OBT)
          ─► notebooks + src/ ─► MLflow (Model Registry, @champion)
          ─► API FastAPI ─► PREDICCIONES / SOLICITUDES ─► monitoreo + alertas ─► tablero
          ─► reentrenamiento con aprobación humana ─► @champion
```

Diagramas: [`docs/arquitectura.png`](proyecto_final/docs/arquitectura.png) · [`docs/modelo_dimensional.png`](proyecto_final/docs/modelo_dimensional.png)

## Estructura

```
proyecto_final/
├── docker-compose.yml          # Postgres, Kestra, Spark, MLflow, API, tablero
├── .env.example                # variables necesarias, sin valores
├── requirements.txt            # entorno local (notebooks y scripts)
├── api/                        # servicio de predicción (FastAPI + Dockerfile)
├── monitoreo/                  # tablero Streamlit (+ Dockerfile)
├── kestra/
│   ├── ingesta.yml             # fuente → RAW → dbt → Spark
│   ├── produccion_lotes.yml    # simulación: envía cada lote semanal a la API
│   ├── monitoreo.yml           # métricas → MONITOREO + alertas (+ dispara reentrenamiento)
│   ├── reentrenamiento.yml     # challenger + pausa de aprobación + promoción
│   ├── rollback_modelo.yml     # vuelve @champion a la versión anterior
│   └── test_conexion_snowflake.yml
├── dbt/                        # Bronze, Silver, Gold y tests
├── spark/build_obt.py          # construye y valida la OBT
├── notebooks/                  # 01–05 modelado · 06 simulación · 07 monitoreo
├── src/
│   ├── datos.py                # carga de la OBT (snapshot)
│   ├── pipeline.py             # Pipeline de scikit-learn y métricas
│   ├── preparar_api.py         # umbral k = 2, línea base y fixtures de CI
│   ├── preparar_simulacion.py  # 9 lotes semanales + 1 lote con deriva
│   ├── cargar_lotes.py         # sube los lotes a PRODUCCION.LOTES_SIMULACION
│   ├── monitoreo.py            # deriva (PSI, KS, chi²), alertas y métricas por lote
│   ├── reentrenar.py           # challenger, comparación y regla de promoción
│   ├── valor_negocio.py        # costo por política
│   ├── analisis_final.py       # cifras por lote y por subgrupo
│   └── exportar_evidencia.py   # CSV de evidencia en docs/
├── tests/                      # contrato, modelo y comportamiento (pytest)
├── data/                       # fuente CSV, snapshot de la OBT, lotes de simulación
└── docs/                       # cambios.md, evidencia CSV, figuras, memos
```

> CI/CD está en la raíz del repositorio (`.github/workflows/`), porque GitHub Actions solo lee esa carpeta.
>
> La lógica de deriva y alertas vive en `src/monitoreo.py` (no en `monitoreo/deriva.py` y `monitoreo/alertas.py`) porque Kestra la ejecuta clonando el repositorio.

## Requisitos

- Docker Desktop (≥ 4,5 GB de memoria)
- Python 3.12
- Cuenta de Snowflake con autenticación por par de llaves RSA

## Instalación

```bash
cd proyecto_final
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

```bash
cp .env.example .env     # completar los valores
mkdir keys               # copiar aquí rsa_key.p8 (no se sube)
```

| Variable | Uso |
|---|---|
| `SNOWFLAKE_*` | conexión de la API, los scripts y el tablero |
| `SNOWFLAKE_PRIVATE_KEY_PATH` | ruta a `keys/rsa_key.p8` |
| `ENV_SNOWFLAKE_*` | la misma conexión, leída por Kestra con `{{ envs.* }}` |
| `SECRET_SNOWFLAKE_PRIVATE_KEY` | llave en base64 para `{{ secret('SNOWFLAKE_PRIVATE_KEY') }}` |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` | servidor y experimento de MLflow |
| `POSTGRES_PASSWORD`, `KESTRA_ADMIN_PASSWORD` | infraestructura local |

En Snowflake, ejecutar una vez [`docs/snowflake_setup.sql`](proyecto_final/docs/snowflake_setup.sql) (base de datos, esquemas y rol).

## Levantar la infraestructura

```bash
docker build -f api/Dockerfile -t api-merma:local .
docker build -f monitoreo/Dockerfile -t tablero-merma:local .
docker tag api-merma:local ghcr.io/0997801643/api-merma:latest
docker tag tablero-merma:local ghcr.io/0997801643/tablero-merma:latest
docker compose up -d
```

Para usar la imagen que publicó el CD para un commit:

```bash
IMAGEN_TAG=<hash> docker compose pull api tablero
IMAGEN_TAG=<hash> docker compose up -d
```

| Servicio | URL |
|---|---|
| Kestra | http://localhost:8080 |
| MLflow | http://localhost:5050 |
| API (docs) | http://localhost:8000/docs |
| Tablero | http://localhost:8501 |

Notas:
- `spark` arranca y verifica la instalación; la OBT la construye Kestra en cada ingesta con la misma imagen. A mano: `docker compose --profile manual run --rm spark-obt`.
- Los flujos de Kestra ejecutan sus scripts de Python con la imagen `api-merma:local`.

## Reproducir desde cero

Todos los comandos se ejecutan desde `proyecto_final/`. Los flujos de Kestra van en el namespace `cerveceria.pset2`.

### 1. Fuente → Raw → Bronze → Silver → Gold → OBT

En Kestra, ejecutar `ingesta_merma_raw`: carga RAW, corre `dbt build` (modelos y tests) y Spark, que valida y publica `CERVECERIA_DB.OBT.OBT_MERMA_LOTES`.

### 2. EDA y modelo

Ejecutar los notebooks `01` a `05` en orden. Registran las corridas en el experimento `merma-cerveza` y el modelo `merma-cerveza-modelo@champion`.

### 3. Registro y API

```bash
python -m src.preparar_api    # umbral k = 2 como tag del champion, línea base y fixtures de CI
python -m pytest tests -v     # 10 pruebas
```

### 4. Simulación

```bash
python -m src.preparar_simulacion    # genera data/simulacion/lotes_simulacion.csv
python -m src.cargar_lotes           # lo sube a PRODUCCION.LOTES_SIMULACION
```

En Kestra, ejecutar `produccion_lotes` con `lotes = todos`. El flujo lee cada lote desde Snowflake (solo las entradas del modelo; la merma la une el monitoreo 2 semanas después) y lo envía a la API.

### 5. Monitoreo

```bash
python -m src.monitoreo --crear-referencia    # una sola vez
```

`monitoreo` se ejecuta solo cuando `produccion_lotes` termina bien (trigger de tipo Flow).

### 6. Reentrenamiento, rollback y respaldo

| Acción | Cómo |
|---|---|
| Reentrenar | Lo lanza `monitoreo` (o a mano: flujo `reentrenamiento`). Se pausa hasta **Resume** con `promover` o `rechazar`. |
| Rollback | Flujo `rollback_modelo`. Comprobar en http://localhost:8000/modelo |
| Respaldo on | `MODO_RESPALDO=1 docker compose up -d api` |
| Respaldo off | `docker compose up -d api` |

### 7. Evidencia

```bash
python -m src.valor_negocio
python -m src.analisis_final
python -m src.exportar_evidencia
```

Luego, Restart y Run All en `notebooks/06_simulacion_produccion.ipynb` y `07_analisis_monitoreo.ipynb`.

## API

| Método | Ruta | Devuelve |
|---|---|---|
| GET | `/salud` | estado del servicio y si el modelo cargó |
| GET | `/modelo` | nombre, versión, alias, `run_id` y umbral |
| POST | `/predecir` | probabilidad, decisión, acción, versión e id de la predicción |
| POST | `/recargar` | fuerza la revisión del alias (rollback inmediato) |

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -H "X-Lote-Simulacion: 1" \
  -d '{"numero_documento": "1301", "estilo": "American Lager",
       "generacion_levadura": 2, "n_cocciones": 2, "coccion_mes": 9,
       "coccion_dia_semana": 3, "carga_planta_14d": 8, "levadura_nueva": 0,
       "sin_generacion": 0, "con_microscopio": 1, "temporada_alta": 1,
       "cepa": "W34/70"}'
```

- El modelo se carga desde `models:/merma-cerveza-modelo@champion`. Un hilo de fondo revisa el alias cada 30 s, así que cambiar de versión no requiere reconstruir la imagen.
- Una entrada inválida devuelve `422` con el campo que falló.
- Cada predicción se guarda en `PRODUCCION.PREDICCIONES`, y cada solicitud (código HTTP, latencia y lote del encabezado `X-Lote-Simulacion`) en `PRODUCCION.SOLICITUDES`.

## Monitoreo y alertas

| Tipo | Métricas | Alerta |
|---|---|---|
| Datos | PSI + KS/chi² por variable, % nulos, % fuera de rango, cepas nuevas | PSI > 0,25 con p < 0,05, solo en variables con peso en el modelo · nulos > 20 % · fuera de rango > 20 % · cualquier cepa nueva |
| Modelo | probabilidad media, % con clarificante, PSI de la probabilidad | PSI > 0,25 con p < 0,05 |
| Desempeño | ROC-AUC por lote, costo por lote (modelo, regla por estilo, sin acción) | ROC-AUC < 0,389 (media CV − 2 desv.) · costo del modelo > regla por estilo |
| Sistema | volumen, latencia p50/p95, errores 4xx/5xx, respaldo, faltantes | p95 > 500 ms · 4xx > 5 % · 5xx > 0 % · cualquier respaldo o faltante |
| Negocio | clarificantes por semana | > 2 por semana |

- 2 lotes etiquetados seguidos con ROC-AUC bajo el límite → `monitoreo` lanza `reentrenamiento`. La deriva de datos sola no reentrena.
- Si un lote se reenvía, solo cuentan las solicitudes de su última ejecución.

## Pruebas

```bash
python -m pytest tests -v
```

| Archivo | Qué verifica |
|---|---|
| `test_contrato.py` | 200 con el esquema esperado; 422 por campo faltante, tipo incorrecto y categoría desconocida; el rechazo queda registrado con su código y su lote |
| `test_modelo.py` | la copia del champion reproduce el ROC-AUC registrado en MLflow (tolerancia 1e-6) |
| `test_comportamiento.py` | reglas de negocio del modelo, respaldo si el modelo no carga e interruptor `MODO_RESPALDO` |

En CI no hay servidor de MLflow: la API usa la copia exportada en `tests/fixtures/modelo`.

## Evidencia (`docs/`)

| Archivo | Contenido |
|---|---|
| `predicciones.csv` | tabla `PREDICCIONES` |
| `solicitudes.csv` | tabla `SOLICITUDES` |
| `monitoreo.csv` | tabla `MONITOREO` |
| `alertas.csv` | alertas con severidad, responsable y acción |
| `corridas.csv` | todas las corridas de MLflow |
| `reentrenamientos.csv` | corridas con `fase = reentrenamiento` |
| `cambios.md` | observaciones de los PSets 1–3 y decisiones actualizadas |
| `figs/` | capturas de Kestra, MLflow, API, GitHub Actions, GHCR y tablero |

## Seguridad

- `.env`, `keys/`, `.venv/` y `registros/` están en `.gitignore`.
- Kestra lee la configuración con `{{ envs.* }}` y la llave con `{{ secret(...) }}`.
- Ningún archivo del repositorio contiene credenciales.
