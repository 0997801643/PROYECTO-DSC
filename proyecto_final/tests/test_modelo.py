"""
Prueba del modelo (sección 3.2 del PSet 4).

La copia exportada del champion (tests/fixtures/modelo) debe reproducir el ROC-AUC registrado
en MLflow sobre la muestra fija guardada en el repositorio (tests/fixtures/muestra_prueba.csv).

Tolerancia: 1e-6. Es el mismo modelo serializado aplicado a los mismos datos, así que el
resultado debe ser idéntico salvo redondeo de punto flotante. Una diferencia mayor significa
que cambió el modelo, la preparación de los datos o una versión de librería, y debe frenar el CI.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from conftest import FIXTURES


# 5. El modelo reproduce la métrica registrada
def test_modelo_reproduce_metrica_registrada():
    import mlflow.sklearn
    info = json.loads((FIXTURES / "info_modelo.json").read_text())
    modelo = mlflow.sklearn.load_model(str(FIXTURES / "modelo"))
    muestra = pd.read_csv(FIXTURES / "muestra_prueba.csv")
    X, y = muestra.drop(columns="merma_alta"), muestra["merma_alta"]
    roc = roc_auc_score(y, modelo.predict_proba(X)[:, 1])
    assert np.isclose(roc, info["roc_auc_registrado"], atol=1e-6), (
        f"ROC-AUC {roc:.6f} no coincide con el registrado {info['roc_auc_registrado']:.6f}")
