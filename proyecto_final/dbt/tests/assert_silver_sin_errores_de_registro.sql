-- Regla de Silver: se excluyen las órdenes con completado > 105 % del planificado (error de tipeo)
-- y la merma negativa pequeña se ajusta a 0. Después de la limpieza, ninguna orden
-- puede tener completado mayor al planificado.
select numero_documento, cantidad_planificada, cantidad_completada
from {{ ref('int_merma_limpia') }}
where cantidad_completada > cantidad_planificada
