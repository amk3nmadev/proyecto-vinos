"""Pruebas mínimas para verificar que el espacio de trabajo es correcto.

Ejecutar con:  python -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, datos, modelos  # noqa: E402


def test_datos_forma():
    X, y, clases = datos.cargar_datos()
    assert X.shape == (178, 13)
    assert len(y) == 178 and len(clases) == 3
    assert not X.isna().any().any()


def test_particion_estratificada():
    X, y, _ = datos.cargar_datos()
    X_tr, X_te, y_tr, y_te = datos.particionar(X, y)
    assert abs(len(X_te) - round(178 * config.TEST_SIZE)) <= 1
    for c in (0, 1, 2):
        esperado = (y == c).mean() * len(y_te)
        assert abs((y_te == c).sum() - esperado) <= 1


def test_pipelines_entrenan():
    X, y, _ = datos.cargar_datos()
    X_tr, X_te, y_tr, y_te = datos.particionar(X, y)
    for pipe in (modelos.pipeline_rf(), modelos.pipeline_knn()):
        assert pipe.fit(X_tr, y_tr).score(X_te, y_te) > 0.85


if __name__ == "__main__":  # ejecución sin pytest
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK", nombre)
