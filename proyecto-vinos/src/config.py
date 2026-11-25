"""Parámetros del experimento (un único lugar para garantizar reproducibilidad)."""
from pathlib import Path

SEMILLA = 42
TEST_SIZE = 0.20          # partición 80/20 estratificada
CV_FOLDS = 5              # validación cruzada para el ajuste de hiperparámetros
REPETICIONES_SIG = 10     # repeticiones 10x(5-fold) para la prueba de significancia

RAIZ = Path(__file__).resolve().parent.parent
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"

# Rejillas de hiperparámetros
REJILLA_RF = {
    "clf__n_estimators": [50, 100, 200, 500],
    "clf__max_depth": [None, 3, 5, 10],
    "clf__min_samples_split": [2, 5],
    "clf__max_features": ["sqrt", "log2"],
}
REJILLA_KNN = {
    "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
    "clf__metric": ["euclidean", "manhattan", "chebyshev"],
    "clf__weights": ["uniform", "distance"],
    "clf__algorithm": ["auto", "kd_tree", "ball_tree", "brute"],
}
