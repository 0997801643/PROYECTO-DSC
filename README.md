# Proyecto final · Predicción de merma alta en lotes de cerveza

Predice, el día de la cocción, si un lote tendrá **merma de filtración mayor al 5 %**, para decidir a qué lotes aplicar clarificante con un presupuesto de **2 lotes por semana**.

Proyecto end to end del curso *Fundamentos de Ciencia de Datos: End to End* (PSets 1 a 4).
Documento técnico, presentación y video: ver la entrega del PSet 4.

```
SAP (CSV) → Kestra → RAW → dbt (Bronze → Silver → Gold) → Spark (OBT)
         → notebooks + src/ → MLflow (Model Registry @champion)
         → API FastAPI → PREDICCIONES → monitoreo + alertas → tablero
         → reentrenamiento con aprobación humana → @champion
```

## Estructura

```
proyecto_final/
├── docker-compose.yml        # Postgres, Kestra, MLflow, API, tablero (+ dbt y Spark en el perfil "manual")
├── .env.example              # variables necesarias, sin valores
├── requirements.txt          # entorno local (notebooks y scripts)
├── api/                      # servicio de predicción (FastAPI + Dockerfile)
├── monitoreo/                # tablero Streamlit (+ Dockerfile)
├── kestra/
│   ├── ingesta.yml           # fuente → RAW → dbt → Spark (días 1 y 15)
│   ├── produccion_lotes.yml  # simulación: envía cada lote semanal a la API
│   ├── monitoreo.yml         # métricas → PRODUCCION.MONITOREO + alertas
│   ├── reentrenamiento.yml   # challenger + pausa de aprobación + promoción
│   ├── rollback_modelo.yml   # vuelve @champion a la versión anterior
│   └── test_conexion_snowflake.yml
├── dbt/                      # Bronze, Silver, Gold y tests
├── spark/build_obt.py        # construye y valida la OBT
├── notebooks/                # EDA, features, candidatos, tuning, selección (PSet 3)
├── src/
│   ├── datos.py, pipeline.py           # carga de la OBT y Pipeline de scikit-learn
│   ├── preparar_api.py                 # umbral k = 2, línea base y fixtures de CI
│   ├── preparar_simulacion.py          # 9 lotes semanales + 1 lote con deriva
│   ├── monitoreo.py                    # métricas de datos, modelo, desempeño, sistema y negocio
│   ├── reentrenar.py                   # challenger, comparación y regla de promoción
│   ├── valor_negocio.py, analisis_final.py   # cifras de las secciones 8 y 9
├── tests/                    # pytest (contrato, modelo, negocio, respaldo)
├── data/                     # fuente CSV, snapshot de la OBT, lotes de simulación
└── docs/                     # cambios.md, memos anteriores, SQL de calidad, figuras
```
CI y CD viven en la raíz del repositorio: `.github/workflows/ci.yml` y `cd.yml`.

## Requisitos

- Docker Desktop (mínimo ~4,5 GB de memoria asignada).
- Python 3.12 con un entorno virtual (`uv venv` o `python -m venv .venv`).
- Una cuenta de Snowflake con autenticación por llave (par RSA).

## Reproducir desde cero

Todos los comandos se corren desde `proyecto_final/`.

**0. Configuración**
```bash
cp .env.example .env            # completar las variables de Snowflake
mkdir keys                      # copiar aquí rsa_key.p8 (no se sube a GitHub)
uv pip install -r requirements.txt
```
Ejecutar `docs/snowflake_setup.sql` en Snowflake (base de datos, esquemas, rol).

**1. Levantar la infraestructura**
```bash
docker build -f api/Dockerfile -t api-merma:local .
docker build -f monitoreo/Dockerfile -t tablero-merma:local .
docker compose up -d
```
Kestra: http://localhost:8080 · MLflow: http://localhost:5050 · API: http://localhost:8000/docs · Tablero: http://localhost:8501

Las imágenes también se publican en GHCR en cada push a `main` con CI en verde:
`ghcr.io/0997801643/api-merma:<commit>` y `ghcr.io/0997801643/tablero-merma:<commit>`.

**2. Fuente → Raw → Bronze → Silver → Gold → OBT**
En Kestra, crear los flujos de `kestra/` (namespace `cerveceria.pset2`) y ejecutar `ingesta_merma_raw`. El flujo carga RAW, corre `dbt build` (modelos + tests) y Spark, que valida y publica `CERVECERIA_DB.OBT.OBT_MERMA_LOTES`.

**3. EDA y modelo**
Ejecutar los notebooks `01` a `05` en orden. Registran las corridas en el experimento `merma-cerveza` de MLflow y el modelo `merma-cerveza-modelo` con el alias `@champion`.

**4. Registro y API**
```bash
python -m src.preparar_api      # umbral k = 2 como tag del champion, línea base y fixtures de CI
python -m pytest tests -v       # 9 pruebas
```
La API carga `models:/merma-cerveza-modelo@champion` sin reconstruir la imagen.

**5. Simulación**
```bash
python -m src.preparar_simulacion
```
En Kestra, ejecutar `produccion_lotes` (lotes = `todos`). Las predicciones quedan en `PRODUCCION.PREDICCIONES`.

**6. Monitoreo**
```bash
python -m src.monitoreo --crear-referencia     # una vez: distribución del entrenamiento
```
En Kestra, ejecutar `monitoreo`. Las métricas quedan en `PRODUCCION.MONITOREO` y las alertas en el log; el tablero las muestra en http://localhost:8501.

**7. Reentrenamiento, rollback y respaldo**
- Kestra → `reentrenamiento` (input `motivo`): entrena el challenger, se pausa y espera **Resume** con "promover" o "rechazar".
- Kestra → `rollback_modelo`: devuelve `@champion` a la versión anterior; comprobar en http://localhost:8000/modelo.
- Interruptor de respaldo: `MODO_RESPALDO=1 docker compose up -d api` (volver: `docker compose up -d api`).

**8. Cifras del documento**
```bash
python -m src.valor_negocio
python -m src.analisis_final
```

## Seguridad

`.env`, `keys/`, `.venv/` y `registros/` están en `.gitignore`. Los flujos de Kestra leen la configuración con `{{ envs.* }}` y la llave con `{{ secret('SNOWFLAKE_PRIVATE_KEY') }}`; ningún archivo del repositorio contiene credenciales.
