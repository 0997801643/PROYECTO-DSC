# Cambios

Observaciones recibidas en los PSets 1 a 3 y decisiones que cambiaron respecto de un PSet anterior.
"Documento técnico" es el memo del PSet 4.

- [Observaciones del PSet 1](#pset-1)
- [Observaciones del PSet 2](#pset-2)
- [Observaciones del PSet 3](#pset-3)
- [Decisiones actualizadas](#decisiones-actualizadas)

---

## Observaciones

### PSet 1

#### 1.1 · Comparación y selección (9)
- **Observación:** revisar el 18/18 de la Idea 2: el acceso al historial depende de la gerencia y los registros del proceso aún deben recolectarse.
- **Cambio:** el riesgo se materializó en parte. El ERP solo tiene datos desde febrero de 2026 y no existen registros de proceso (temperatura, turbidez). El acceso real fue Medio, no Alto. Se documenta como limitación: 221 lotes en 10 meses.
- **Archivos:** documento técnico, secciones 3 y 10.

#### 1.2 · Problema y formulación (7.5)
- **Observación:** cuantificar los lotes que superan el 5 % de merma y los litros perdidos.
- **Cambio:** 149 de 221 lotes (67 %) superan el 5 %. En las 9 semanas de la simulación, no hacer nada cuesta 4.272 USD (94,93 USD por lote).
- **Archivos:** documento técnico, secciones 1 y 9.

#### 1.3 · Problema y formulación (7.5)
- **Observación:** justificar el umbral del 70 % con los costos de intervención y la meta de recall del 80 %.
- **Cambio:** el 70 % se reemplazó. PSet 3: umbral por costo (0,05). PSet 4: umbral por presupuesto de clarificante, k = 2 lotes por semana (0,6216). Ver [Umbral de decisión](#umbral-de-decisión).
- **Archivos:** `src/preparar_api.py`; documento técnico, sección 1.

#### 1.4 · Problema y formulación (7.5)
- **Observación:** eliminar el resultado esperado de la Idea 1 ("abastecimiento oportuno de clarificantes").
- **Cambio:** el único resultado esperado es el de la Idea 2: menos costo de merma por lote, neto del clarificante innecesario.
- **Archivos:** documento técnico, sección 1.

#### 1.5 · Viabilidad técnica y de datos (8)
- **Observación:** unificar el período de datos (seis meses en el memo, dos a tres años en el pitch).
- **Cambio:** un solo período, febrero a octubre de 2026, porque el ERP solo tiene datos desde febrero.
- **Archivos:** documento técnico, sección 1; presentación.

#### 1.6 · Viabilidad técnica y de datos (8)
- **Observación:** comprobar que existan suficientes lotes con merma > 5 %.
- **Cambio:** hay 149. La clase "alta" es mayoritaria (67 %); por eso la métrica técnica es ROC-AUC.
- **Archivos:** documento técnico, secciones 1 y 4.

#### 1.7 · Viabilidad de negocio (8.5)
- **Observación:** la comparación con períodos sin alertas debe controlar estilo, insumos y temporada.
- **Cambio:** cada política se evalúa sobre los mismos lotes (mismo estilo, insumos y semana) con su merma real. Además se reporta por cepa.
- **Archivos:** `src/valor_negocio.py`, `src/analisis_final.py`; documento técnico, secciones 8 y 9.

#### 1.8 · Alcance del semestre (9.5)
- **Observación:** definir criterios de aceptación del flujo semanal en Kestra.
- **Cambio:**
  - `ingesta`: termina sin error, 0 duplicados y la OBT valida filas = hechos.
  - `produccion_lotes`: una predicción por lote (alerta si falta alguna).
  - `monitoreo`: guarda todas las métricas del lote en `MONITOREO`.
- **Archivos:** `kestra/*.yml`, `src/monitoreo.py`; documento técnico, secciones 3 y 7.

#### 1.9 · Pitch de negocio (6.5)
- **Observación:** eliminar los textos pendientes de la plantilla (`[Apellido 1]`, `[Agregar dato real…]`).
- **Cambio:** la presentación del PSet 4 se hizo desde cero.
- **Archivos:** presentación del PSet 4.

#### 1.10 · Pitch de negocio (6.5)
- **Observación:** incluir el % de lotes con merma > 5 % o los litros perdidos por mes.
- **Cambio:** la presentación incluye el 67 % de lotes con merma alta y el costo semanal de no hacer nada.
- **Archivos:** presentación del PSet 4.

### PSet 2

#### 2.1 · Kestra (7)
- **Observación:** agregar un bloque `errors`.
- **Cambio:** tarea `alerta_falla_ingesta`, que registra en el log (nivel ERROR) el flujo, la ejecución y la hora cuando una tarea falla tras 3 reintentos. Probado con una URL inexistente.
- **Archivos:** `kestra/ingesta.yml`.

#### 2.2 · Kestra (7)
- **Observación:** integrar dbt y Spark en Kestra; se ejecutaban a mano.
- **Cambio:** el flujo corre de punta a punta: RAW → `dbt build` → Spark (OBT). Una tarea `WorkingDirectory` clona el repositorio y ejecuta dbt y Spark en contenedores con las mismas imágenes del compose. Evidencia: 5 modelos y 28 tests en verde (PASS=33), OBT de 221 filas validada.
- **Archivos:** `kestra/ingesta.yml`.

#### 2.3 · Calidad de datos (9)
- **Observación:** determinar qué representan las dos mermas negativas y justificar su reemplazo por cero.
- **Cambio:** revisadas en SAP: errores de tipeo por lotes mezclados. Como el volumen real no se puede reconstruir, el completado se ajusta al planificado (merma 0 %) y se conservan los 2 lotes.
- **Archivos:** `dbt/models/intermedia/int_merma_limpia.sql`; documento técnico, sección 3.

#### 2.4 · dbt (7)
- **Observación:** agregar tests de las reglas de Silver, sobre todo órdenes abiertas y productos sin levadura.
- **Cambio:** 3 tests singulares (de 25 a 28, todos en verde):
  - sin órdenes abiertas (fecha de cierre nula o completado ≤ 0);
  - sin productos sin levadura (Platonic, Turbo + Guaraná o cepa nula);
  - sin errores de registro (completado > planificado).
- **Archivos:** `dbt/tests/assert_silver_sin_ordenes_abiertas.sql`, `dbt/tests/assert_silver_sin_productos_sin_levadura.sql`, `dbt/tests/assert_silver_sin_errores_de_registro.sql`.

#### 2.5 · Star schema (7.5)
- **Observación:** ampliar la justificación del modelo dimensional con tanques, cepas y generaciones de levadura.
- **Cambio:** la cepa queda en `dim_estilo` porque en esta planta depende del estilo; la generación y el tanque cambian lote a lote, así que son atributos del hecho.
- **Archivos:** documento técnico, sección 3.

#### 2.6 · Spark y OBT (8.5)
- **Observación:** documentar mejor la conciliación antes de publicar la OBT.
- **Cambio:** tres validaciones (221 filas en hechos = 221 en la OBT, 221 órdenes únicas, 0 filas sin estilo o fecha). Si alguna falla, el script se detiene sin escribir y conserva la última OBT válida.
- **Archivos:** `spark/build_obt.py`; documento técnico, sección 3.

#### 2.7 · Infraestructura (8)
- **Observación:** quitar la cuenta de Snowflake escrita en los flows.
- **Cambio:** cuenta, usuario, rol, warehouse y base de datos pasan al `.env` como `ENV_SNOWFLAKE_*` (`{{ envs.* }}`); la llave sigue como secreto. `.env.example` documenta las variables sin valores.
- **Archivos:** `kestra/ingesta.yml`, `kestra/test_conexion_snowflake.yml`, `.env.example`.

#### 2.8 · Documento técnico (6.5)
- **Observación:** eliminar los textos pendientes de la plantilla.
- **Cambio:** el documento técnico del PSet 4 se escribió desde cero.
- **Archivos:** documento técnico.

#### 2.9 · Documento técnico (6.5)
- **Observación:** completar la arquitectura, comparar estrella vs. OBT y explicar qué cambiaría con streaming.
- **Cambio:** diagrama completo (fuente → API → monitoreo → reentrenamiento); comparación estrella (análisis) vs. OBT (modelo); streaming convendría con sensores durante el filtrado.
- **Archivos:** documento técnico, secciones 2 y 3.

### PSet 3

Calificación 100/100, sin observaciones.

---

## Decisiones actualizadas

### Horizonte
- **Antes:** al terminar la fermentación (PSet 1).
- **Ahora:** día de la cocción (PSets 3 y 4).
- **Por qué:** deja de 3 a 5 días para conseguir y dosificar el clarificante, y ninguna feature usa información posterior a la cocción.
- **Archivos:** `src/pipeline.py`; documento técnico, sección 1.

### Umbral de decisión
- **Antes:** probabilidad > 70 % (PSet 1) → 0,05 por costo (PSet 3).
- **Ahora:** presupuesto de k = 2 lotes por semana → umbral 0,6216, guardado como tag `umbral_decision` de la versión del modelo.
- **Por qué:** 0,05 equivalía a clarificar casi todos los lotes. En planta el clarificante tiene presupuesto semanal; k = 1 era muy restrictivo para ~6 cocciones por semana.
- **Archivos:** `src/preparar_api.py`, `api/main.py`, `src/reentrenar.py`.

### Línea base de negocio
- **Antes:** "siempre clarificante" y "nunca clarificante" (PSet 3).
- **Ahora:** regla por estilo (clarificar Belgian Tripel, Honey Bock, Iguana, Oktoberfest y Red Llama: 23 % de los lotes) y "no hacer nada".
- **Por qué:** es la regla que aplicaría un jefe de producción con el mismo presupuesto. Es por estilo y no por cepa porque la cepa es una feature del modelo y el estilo es como la planta organiza la producción.
- **Archivos:** `src/preparar_api.py`, `api/linea_base.json`.

### Métrica técnica
- **Antes:** recall (meta 80 %) y precisión (PSet 1).
- **Ahora:** ROC-AUC (PSet 3).
- **Por qué:** con 67 % de lotes altos, "siempre alta" logra F1 0,79 sin aprender. ROC-AUC mide si el modelo ordena bien el riesgo, que es lo que usa el presupuesto k = 2.
- **Archivos:** `src/pipeline.py`.

### Métrica de negocio
- **Antes:** merma promedio y litros perdidos por mes (PSet 1).
- **Ahora:** costo de errores por lote en USD (PSet 3).
- **Por qué:** pone en la misma unidad los dos errores: cerveza perdida y clarificante desperdiciado.
- **Archivos:** `src/pipeline.py`, `src/monitoreo.py`, `src/valor_negocio.py`.

### Período de datos
- **Antes:** 6 meses (memo) / 2–3 años (pitch) (PSet 1).
- **Ahora:** febrero–octubre 2026.
- **Por qué:** el ERP solo tiene datos desde febrero.
- **Archivos:** documento técnico, sección 1.

### Regla de promoción
- **Antes:** no existía.
- **Ahora:** el challenger se recomienda si ahorra más de 1 desv. estándar del costo en la CV del champion (20,87 USD por lote) y no pierde más de 1 desv. de ROC-AUC (0,123). La decisión final es humana.
- **Por qué:** con 221 lotes, una mejora menor que la variación entre pliegues es ruido.
- **Archivos:** `src/reentrenar.py`, `kestra/reentrenamiento.yml`.

### Estructura del repositorio
- **Antes:** PSet 2 en la raíz y `pset_3/` en el repositorio `PSET2`.
- **Ahora:** todo en `proyecto_final/`, en el repositorio `PROYECTO-DSC`.
- **Por qué:** el PSet 4 exige un solo proyecto con un solo `docker-compose.yml`. Se unieron los dos compose y los dos `.env`, conservando los volúmenes (Model Registry y flujos de Kestra).
- **Archivos:** `docker-compose.yml`, `src/datos.py`, `.gitignore`, `kestra/ingesta.yml` (URL del CSV).

### Alertas de deriva de datos
- **Antes:** PSI > 0,25 con KS/chi² p < 0,05 en todas las variables salvo las de calendario (31 alertas en la primera versión).
- **Ahora:** el PSI alerta solo en las variables con peso en el modelo (`generacion_levadura`). La deriva de las demás se mide sin alertar. Nulos, fuera de rango y cepas nuevas alertan en todas.
- **Por qué:** `con_microscopio` alertaba en los 10 lotes (el microscopio se compró en agosto: 171 de 176 lotes de entrenamiento sin él), pero su coeficiente es 0 y su deriva no cambia ninguna predicción. Lo mismo con `carga_planta_14d`; su cambio en el lote con deriva lo detecta la alerta de fuera de rango. Verificado en el notebook 07 (tabla de coeficientes).
- **Archivos:** `src/monitoreo.py`, `notebooks/07_analisis_monitoreo.ipynb`.

### Disparo del reentrenamiento
- **Antes:** manual o por calendario mensual.
- **Ahora:** por métrica. `monitoreo` lanza `reentrenamiento` cuando el ROC-AUC de 2 lotes etiquetados seguidos queda bajo media CV − 2 desv. (0,389). La deriva de datos sola no dispara.
- **Por qué:** reentrenar solo tiene sentido si cae el desempeño.
- **Archivos:** `src/monitoreo.py`, `kestra/monitoreo.yml`.

### Ejecución del monitoreo
- **Antes:** a mano después de cada lote.
- **Ahora:** trigger de tipo Flow (`dependsOn`) cada vez que `produccion_lotes` termina bien.
- **Por qué:** ningún lote queda sin monitorear.
- **Archivos:** `kestra/monitoreo.yml`.

### Imágenes del compose
- **Antes:** `api-merma:local` y `tablero-merma:local`, construidas en local.
- **Ahora:** `ghcr.io/0997801643/api-merma:${IMAGEN_TAG}` y `tablero-merma:${IMAGEN_TAG}`, publicadas por el CD con el hash del commit.
- **Por qué:** el compose debe usar la imagen publicada. Kestra sigue usando `api-merma:local` para sus scripts de Python.
- **Archivos:** `docker-compose.yml`, `.github/workflows/cd.yml`.

### Métricas de sistema
- **Antes:** latencia p50/p95, respaldo y predicciones faltantes, calculadas desde `PREDICCIONES`.
- **Ahora:** además, volumen de solicitudes y tasa de errores 4xx (alerta > 5 %) y 5xx (alerta > 0 %) por lote. La API guarda cada solicitud en `PRODUCCION.SOLICITUDES` (ruta, código, latencia y lote del encabezado `X-Lote-Simulacion`).
- **Por qué:** una solicitud rechazada nunca llega a `PREDICCIONES`. Un 422 ocasional es normal (error de digitación); más de 1 de cada 20 indica un cambio en el envío o en el esquema. Un 500 nunca debería ocurrir.
- **Archivos:** `api/registro.py`, `api/main.py`, `kestra/produccion_lotes.yml`, `src/monitoreo.py`, `src/exportar_evidencia.py`, `monitoreo/tablero.py`, `tests/test_contrato.py`.

### Revisión del alias en la API
- **Antes:** cada 30 s, dentro de la primera solicitud que llegaba (su latencia incluía la consulta a MLflow).
- **Ahora:** un hilo de fondo revisa el alias cada 30 s; las predicciones nunca consultan MLflow.
- **Por qué:** en la primera simulación hubo predicciones aisladas de 2 a 8 s con medianas por lote de 18 a 95 ms; esto elimina la causa más probable. `POST /recargar` sigue aplicando un rollback al instante.
- **Archivos:** `api/main.py`.

### Severidad de `roc_auc_lote`
- **Antes:** sin plantilla; sus alertas salían sin severidad, responsable ni acción.
- **Ahora:** Alta · analista de datos · si se repite 2 lotes seguidos, el monitoreo lanza el reentrenamiento con aprobación humana.
- **Por qué:** es la alerta que dispara el reentrenamiento.
- **Archivos:** `src/exportar_evidencia.py`.

### Tablero
- **Antes:** alertas y métricas por lote.
- **Ahora:** además, los reentrenamientos (leídos de MLflow) con el lote que los disparó, el ROC-AUC por lote frente al límite de la CV, y los errores 4xx/5xx y el volumen por lote.
- **Por qué:** el PDF pide que el tablero muestre alertas y reentrenamientos.
- **Archivos:** `monitoreo/tablero.py`, `docker-compose.yml`.
