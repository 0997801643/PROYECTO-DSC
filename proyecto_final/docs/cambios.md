# Registro de cambios del proyecto

Cada observación recibida en los PSets #1 a #3 y cada decisión que cambió respecto de un PSet anterior, con lo que se modificó y en qué archivo.

## 1. Observaciones del profesor

### PSet #1

| # | Criterio (nota) | Observación | Qué cambié | Archivo(s) |
|---|---|---|---|---|
| 1.1 | Comparación y selección (9) | Revisar el 18/18 de la Idea 2: el acceso al historial depende de la gerencia y los registros del proceso todavía deben recolectarse. | | |
| 1.2 | Problema y formulación (7.5) | Cuantificar los lotes que superan el 5 % de merma y los litros perdidos. | | |
| 1.3 | Problema y formulación (7.5) | Justificar el umbral del 70 % con los costos de intervención y la meta de recall del 80 %. | | |
| 1.4 | Problema y formulación (7.5) | Eliminar el resultado esperado de la Idea 1 ("abastecimiento oportuno de clarificantes"). | | |
| 1.5 | Viabilidad técnica y de datos (8) | Unificar el período de datos: seis meses en el memo y de dos a tres años en el pitch. | | |
| 1.6 | Viabilidad técnica y de datos (8) | Comprobar que existan suficientes lotes con merma superior al 5 %. | | |
| 1.7 | Viabilidad de negocio (8.5) | La comparación con períodos sin alertas debe controlar estilo, insumos y temporada. | | |
| 1.8 | Alcance del semestre (9.5) | Definir criterios de aceptación del flujo semanal en Kestra. | | |
| 1.9 | Pitch de negocio (6.5) | Eliminar los textos pendientes de la plantilla ([Apellido 1], [Agregar dato real…]). | | |
| 1.10 | Pitch de negocio (6.5) | Incluir el % de lotes con merma > 5 % o los litros perdidos por mes. | | |

### PSet #2

| # | Criterio (nota) | Observación | Qué cambié | Archivo(s) |
|---|---|---|---|---|
| 2.1 | Kestra (7) | Agregar un bloque de errores (`errors`). | | |
| 2.2 | Kestra (7) | Integrar dbt y Spark en Kestra; hoy ambas etapas se ejecutan a mano. | | |
| 2.3 | Calidad de datos (9) | Justificar por qué las dos mermas negativas se reemplazan por cero; antes, determinar qué representan esos valores. | | |
| 2.4 | dbt (7) | Agregar tests que verifiquen las reglas de Silver, sobre todo la exclusión de órdenes abiertas y de productos sin levadura. | | |
| 2.5 | Star schema (7.5) | Ampliar la justificación del modelo dimensional considerando tanques, cepas y generaciones de levadura. | | |
| 2.6 | Spark y OBT (8.5) | Documentar mejor la conciliación antes de publicar la OBT. | | |
| 2.7 | Infraestructura (8) | Quitar la referencia fija a la cuenta de Snowflake en los flows y pasarla a la configuración. | | |
| 2.8 | Documento técnico (6.5) | Eliminar los textos pendientes de la plantilla. | | |
| 2.9 | Documento técnico (6.5) | Completar la arquitectura, comparar estrella vs. OBT y explicar qué cambiaría para usar streaming. | | |

### PSet #3

_Pendiente: aún no se recibe la retroalimentación._

| # | Criterio (nota) | Observación | Qué cambié | Archivo(s) |
|---|---|---|---|---|

## 2. Decisiones actualizadas

| Decisión | Antes (PSet) | Ahora | Por qué | Archivos actualizados |
|---|---|---|---|---|
| Horizonte | Al terminar la fermentación (PSet #1) | Día de la cocción (PSet #3 y #4) | | |
| Umbral de decisión | Probabilidad > 70 % (PSet #1) → 0,05 por costo (PSet #3) | Ranking: clarificante a los k = 2 lotes de mayor riesgo por semana (supuesto de capacidad: 1–2 lotes/semana; k = 1 como sensibilidad) | | |
| Línea base de negocio | "Siempre clarificante" y "nunca clarificante" (PSet #3) | Con k = 2: regla "los 2 lotes de la cepa con más merma histórica" y "no hacer nada" | | |
| Métrica técnica | Recall (meta 80 %) y precisión (PSet #1) | ROC-AUC (PSet #3) | | |
| Métrica de negocio | Merma promedio y litros perdidos por mes (PSet #1) | Costo de errores por lote (USD) (PSet #3) | | |
| Período de datos | 6 meses (memo) / 2–3 años (pitch) (PSet #1) | Febrero–octubre 2026: el ERP solo tiene datos desde febrero | | |
| Estructura del repo | `pset_2` en la raíz y `pset_3/` | Todo en `proyecto_final/` | | |
