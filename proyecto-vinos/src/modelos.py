"""Definición de los pipelines de los dos algoritmos comparados."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def pipeline_rf():
    # El escalado no afecta a los árboles, pero se mantiene el mismo preproceso
    # en ambos modelos para que la comparación sea justa.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=config.SEMILLA, n_jobs=-1)),
    ])


def pipeline_knn():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])
