-- Regla de Silver: se excluyen las órdenes abiertas (sin fecha de cierre o sin cantidad completada).
-- El test falla si alguna orden abierta llegó a Silver.
select numero_documento, fecha_cierre, cantidad_completada
from {{ ref('int_merma_limpia') }}
where fecha_cierre is null
   or cantidad_completada is null
   or cantidad_completada <= 0