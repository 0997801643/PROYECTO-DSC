# PSet 3 · Experimentación y selección del modelo: merma alta de cerveza

Predice si un lote de producción tendrá **merma alta (> 5 % del volumen planificado)** para decidir si se aplica clarificante.
Parte de la OBT `CERVECERIA_DB.OBT.OBT_MERMA_LOTES` construida en el PSet 2.

```
OBT → EDA → Features → Pipeline → Línea base → Candidatos → Tuning → Selección → Model Registry
```

## Estructura

```
pset_3/
├── docker-compose.yml        # servidor de MLflow (backend SQLite + artifact store)
├── .env.example              # variables de MLflow (sin secretos)
├── requirements.txt
├── data/obt_snapshot.parquet # snapshot fijo de la OBT (mismo hash en todos los notebooks)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb
│   ├── 04_tuning.ipynb
│   └── 05_seleccion.ipynb
├── src/
│   ├── datos.py              # conexión a Snowflake, snapshot y metadatos de los datos
│   └── pipeline.py           # features, división, Pipeline/ColumnTransformer, métricas y registro en MLflow
└── docs/
    ├── corridas.csv          # todas las corridas del experimento (mlflow.search_runs)
    ├── figs/                 # gráficos usados en el memo
    └── PSet3_memo_Velastegui_Diaz.pdf
```

## 1. Levantar MLflow

```bash
cd pset_3
cp .env.example .env
docker compose up -d
```

Interfaz en <http://localhost:5050> (se usa 5050 porque en macOS el 5000 lo ocupa AirPlay).
Backend store: SQLite en el volumen `datos_mlflow`; artifact store: `/mlflow/artifacts` en el mismo volumen, servido por HTTP (`--serve-artifacts`), así los notebooks pueden subir modelos desde la laptop.

## 2. Entorno de Python (3.12)

```bash
# desde la raíz del repo
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r pset_3/requirements.txt
```

En VS Code, elegir el kernel `.venv (Python 3.12)` en cada notebook.

## 3. Conexión a Snowflake

Se reutiliza la configuración del PSet 2: el `.env` de la **raíz** del repo (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_PRIVATE_KEY_PATH=keys/rsa_key.p8`) y la llave privada en `keys/`.
`src/datos.py` lee la OBT una vez y guarda `data/obt_snapshot.parquet`; los notebooks leen ese snapshot. Para volver a consultar Snowflake: `cargar_obt(forzar=True)`.

## 4. LLM local (notebook 03)

La prueba con LLM usa [Ollama](https://ollama.com) en `http://localhost:11434`:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

## 5. Orden de ejecución

Con MLflow y Ollama encendidos, ejecutar cada notebook con **Restart → Run All**, en este orden:

| Notebook | Qué hace | Corridas en MLflow |
|---|---|---|
| `01_eda` | Calidad de datos, target, outliers, correlación y hallazgos | — |
| `02_features` | Features derivadas, prueba de no fuga, división y esquema de CV | — |
| `03_baseline_candidatos` | 2 líneas base, 5 familias de modelos y LLM (~6 min) | 8 |
| `04_tuning` | Búsqueda aleatoria de 30 configuraciones × 3 modelos (corridas anidadas) | 93 |
| `05_seleccion` | Umbral por costo, prueba final, Model Registry, reproducción y `docs/corridas.csv` | 2 |

Modelo registrado: `merma-cerveza-modelo` con alias `@champion` (regresión logística) y `@challenger` (MLP).

## Seguridad

No se suben credenciales: `.env`, `keys/` y `.venv/` están en `.gitignore`.
