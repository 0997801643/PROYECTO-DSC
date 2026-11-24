-- Regla de Silver: se excluyen los productos que no son cerveza (no usan levadura).
-- El test falla si alguno llegó a Silver, por nombre o por no tener cepa.
select numero_documento, estilo, cepa
from {{ ref('int_merma_limpia') }}
where estilo in ('Platonic', 'Turbo + Guarana Tropical')
   or cepa is null
   