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
├── docker-compose.yml        # Postgres, Kestra, Spark, MLflow, API y tablero (+ dbt y spark-obt en el perfil "manual")
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
├── notebooks/                # 01–05 (PSet 3), 06 simulación y 07 monitoreo (PSet 4)
├── src/
│   ├── datos.py, pipeline.py           # carga de la OBT y Pipeline de scikit-learn
│   ├── preparar_api.py                 # umbral k = 2, línea base y fixtures de CI
│   ├── preparar_simulacion.py          # 9 lotes semanales + 1 lote con deriva
│   ├── monitoreo.py                    # métricas de datos, modelo, desempeño, sistema y negocio
│   ├── reentrenar.py                   # challenger, comparación y regla de promoción
│   ├── valor_negocio.py, analisis_final.py   # cifras de las secciones 8 y 9
│   ├── exportar_evidencia.py           # CSV de evidencia en docs/
├── tests/                    # test_contrato.py, test_modelo.py, test_comportamiento.py
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

En `.env` también van `POSTGRES_PASSWORD` y `KESTRA_ADMIN_PASSWORD` (ver `.env.example`).
```bash
docker build -f api/Dockerfile -t api-merma:local .
docker build -f monitoreo/Dockerfile -t tablero-merma:local .
docker tag api-merma:local ghcr.io/0997801643/api-merma:latest
docker tag tablero-merma:local ghcr.io/0997801643/tablero-merma:latest
docker compose up -d
```
El compose levanta Postgres, Kestra, Spark, MLflow, la API y el tablero. La API y el tablero usan las imágenes que publica el CD en GHCR con el tag `IMAGEN_TAG`, que es el hash del commit (por defecto `latest`): `IMAGEN_TAG=<hash> docker compose pull` y luego `IMAGEN_TAG=<hash> docker compose up -d`. Los `docker tag` de arriba sirven para usar las imágenes construidas en local. Spark se levanta y verifica su instalación; el trabajo que construye la OBT lo ejecuta Kestra en cada ingesta, con la misma imagen (a mano: `docker compose --profile manual run --rm spark-obt`). Los flujos de Kestra corren sus tareas de Python con la imagen `api-merma:local`.

Kestra: http://localhost:8080 · MLflow: http://localhost:5050 · API: http://localhost:8000/docs · Tablero: http://localhost:8501

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
`monitoreo` corre solo cada vez que termina bien `produccion_lotes` (trigger de tipo Flow); también se puede ejecutar a mano. Las métricas quedan en `PRODUCCION.MONITOREO` y las alertas en el log; el tablero las muestra en http://localhost:8501. Si el ROC-AUC por lote cae bajo media − 2 desv. de la CV en 2 lotes etiquetados seguidos, `monitoreo` lanza `reentrenamiento` con ese motivo.

La lógica de deriva (PSI, KS, chi²) y de alertas vive en `src/monitoreo.py` (en lugar de `monitoreo/deriva.py` y `monitoreo/alertas.py`), porque Kestra la ejecuta clonando el repositorio. La carpeta `monitoreo/` contiene el tablero, que además lee de MLflow los reentrenamientos y muestra el lote que disparó cada uno.

Reglas de alerta de datos: el PSI alerta solo si la prueba KS o chi² confirma el cambio (p < 0,05) y solo en las variables con peso en el modelo (`generacion_levadura`; `temporada_alta` es de calendario). La deriva de las demás, como `con_microscopio`, se mide y se guarda sin alertar. La calidad de datos (nulos, fuera de rango, cepas nuevas) alerta en todas las variables.

**7. Reentrenamiento, rollback y respaldo**
- Kestra → `reentrenamiento` (input `motivo`): entrena el challenger, se pausa y espera **Resume** con "promover" o "rechazar".
- Kestra → `rollback_modelo`: devuelve `@champion` a la versión anterior; comprobar en http://localhost:8000/modelo.
- Interruptor de respaldo: `MODO_RESPALDO=1 docker compose up -d api` (volver: `docker compose up -d api`).

**8. Cifras del documento, evidencia y notebooks de producción**
```bash
python -m src.valor_negocio
python -m src.analisis_final
python -m src.exportar_evidencia      # docs/predicciones, monitoreo, alertas, corridas, reentrenamientos (.csv)
```
Luego ejecutar `notebooks/06_simulacion_produccion.ipynb` y `07_analisis_monitoreo.ipynb` (Restart y Run All).

## Seguridad

`.env`, `keys/`, `.venv/` y `registros/` están en `.gitignore`. Los flujos de Kestra leen la configuración con `{{ envs.* }}` y la llave con `{{ secret('SNOWFLAKE_PRIVATE_KEY') }}`; ningún archivo del repositorio contiene credenciales.
