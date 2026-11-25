#!/usr/bin/env python
"""Ejecuta TODO el experimento: EDA -> ajuste -> evaluación -> significancia.

Uso:  python run_experiment.py
Salida: carpeta resultados/ (tablas CSV/JSON/TXT y figuras PNG).
"""
import json

import pandas as pd

from src import config, datos, modelos, experimento


def main():
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    print("1/5 Carga y EDA...")
    X, y, clases = datos.cargar_datos()
    datos.resumen_eda(X, y, clases)
    X_tr, X_te, y_tr, y_te = datos.particionar(X, y)
    print(f"    entrenamiento={len(X_tr)}  prueba={len(X_te)}  atributos={X.shape[1]}")

    print("2/5 Ajuste de hiperparámetros (rejilla + CV)...")
    rf, info_rf, _ = experimento.ajustar("Random Forest", modelos.pipeline_rf(),
                                         config.REJILLA_RF, X_tr, y_tr)
    knn, info_knn, _ = experimento.ajustar("KNN", modelos.pipeline_knn(),
                                           config.REJILLA_KNN, X_tr, y_tr)
    with open(config.DIR_RESULTADOS / "tuning.json", "w", encoding="utf-8") as f:
        json.dump([info_rf, info_knn], f, indent=2, ensure_ascii=False)

    print("3/5 Evaluación en el conjunto de prueba...")
    filas, reportes = [], []
    for nombre, m, info in [("Random Forest", rf, info_rf), ("KNN", knn, info_knn)]:
        res, txt = experimento.evaluar(nombre, m, X_te, y_te, clases)
        res["f1_macro_cv_train"] = info["f1_macro_cv"]
        res["tiempo_tuning_s"] = info["tiempo_tuning_s"]
        filas.append(res)
        reportes.append(f"=== {nombre} ===\n{txt}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_RESULTADOS / "comparacion.csv", index=False)
    (config.DIR_RESULTADOS / "reporte_clasificacion.txt").write_text(
        "\n".join(reportes), encoding="utf-8")
    print(tabla.to_string(index=False))

    print("4/5 Prueba de significancia (10x5-fold sobre todo el dataset)...")
    sig = experimento.prueba_significancia(rf, knn, X, y)
    sig["nota"] = "A = Random Forest, B = KNN (con los mejores hiperparámetros)"
    with open(config.DIR_RESULTADOS / "significancia.json", "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2, ensure_ascii=False)
    print("   ", sig)

    print("5/5 Importancia de rasgos y curvas de aprendizaje...")
    experimento.importancia_rasgos(rf, X_te, y_te, list(X.columns))
    experimento.curvas_aprendizaje({"Random Forest": rf, "KNN": knn}, X, y)
    print(f"Listo. Resultados en: {config.DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
