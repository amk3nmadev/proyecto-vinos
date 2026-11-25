"""Ajuste, evaluación y comparación estadística de Random Forest vs KNN."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     learning_curve)

from . import config


def ajustar(nombre, pipe, rejilla, X_tr, y_tr):
    """Búsqueda en rejilla con CV estratificada; devuelve (mejor_modelo, info)."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEMILLA)
    gs = GridSearchCV(pipe, rejilla, cv=cv, scoring="f1_macro", n_jobs=-1)
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    t_tuning = time.perf_counter() - t0
    info = {
        "modelo": nombre,
        "mejores_parametros": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "f1_macro_cv": round(float(gs.best_score_), 4),
        "combinaciones_probadas": len(gs.cv_results_["params"]),
        "tiempo_tuning_s": round(t_tuning, 2),
    }
    return gs.best_estimator_, info, gs


def evaluar(nombre, modelo, X_te, y_te, clases):
    t0 = time.perf_counter()
    pred = modelo.predict(X_te)
    t_pred = time.perf_counter() - t0
    res = {
        "modelo": nombre,
        "exactitud": round(accuracy_score(y_te, pred), 4),
        "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
        "tiempo_prediccion_ms": round(t_pred * 1000, 2),
    }
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión – {nombre}")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / f"confusion_{nombre.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)
    informe_txt = classification_report(y_te, pred, target_names=clases, digits=3)
    return res, informe_txt


def prueba_significancia(modelo_a, modelo_b, X, y):
    """Compara dos modelos con 10x(5-fold) repetido sobre TODO el dataset.

    Se reporta el t-test pareado con la corrección de Nadeau & Bengio (2003),
    que tiene en cuenta la dependencia entre particiones solapadas, y el
    test de Wilcoxon como alternativa no paramétrica.
    """
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS,
                                 n_repeats=config.REPETICIONES_SIG,
                                 random_state=config.SEMILLA)
    sa = cross_val_score(clone(modelo_a), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    sb = cross_val_score(clone(modelo_b), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    d = sa - sb
    n = len(d)
    n_te, n_tr = len(X) / config.CV_FOLDS, len(X) * (1 - 1 / config.CV_FOLDS)
    var = d.var(ddof=1)
    if var == 0:
        t_corr, p_corr = 0.0, 1.0
    else:
        t_corr = d.mean() / np.sqrt((1 / n + n_te / n_tr) * var)
        p_corr = 2 * stats.t.sf(abs(t_corr), df=n - 1)
    try:
        p_wil = stats.wilcoxon(sa, sb).pvalue
    except ValueError:  # todas las diferencias son 0
        p_wil = 1.0
    return {
        "exactitud_cv_modelo_A_media": round(float(sa.mean()), 4),
        "exactitud_cv_modelo_A_std": round(float(sa.std()), 4),
        "exactitud_cv_modelo_B_media": round(float(sb.mean()), 4),
        "exactitud_cv_modelo_B_std": round(float(sb.std()), 4),
        "diferencia_media": round(float(d.mean()), 4),
        "t_corregido_nadeau_bengio": round(float(t_corr), 3),
        "p_valor_t_corregido": round(float(p_corr), 4),
        "p_valor_wilcoxon": round(float(p_wil), 4),
        "n_evaluaciones": int(n),
    }


def importancia_rasgos(rf, X_te, y_te, columnas):
    imp_gini = rf.named_steps["clf"].feature_importances_
    perm = permutation_importance(rf, X_te, y_te, n_repeats=30,
                                  random_state=config.SEMILLA, scoring="f1_macro")
    df = pd.DataFrame({
        "atributo": columnas,
        "importancia_gini": imp_gini,
        "importancia_permutacion": perm.importances_mean,
    }).sort_values("importancia_gini", ascending=False).round(4)
    df.to_csv(config.DIR_RESULTADOS / "importancia_rasgos.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(x="importancia_gini", y="atributo", data=df, ax=ax, color="#4c72b0")
    ax.set_title("Importancia de atributos (Random Forest, Gini)")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "importancia_rasgos.png", dpi=150)
    plt.close(fig)
    return df


def curvas_aprendizaje(modelos, X, y):
    """Extensión: curvas de aprendizaje de ambos modelos."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEMILLA)
    fig, ax = plt.subplots(figsize=(6, 4))
    for nombre, m in modelos.items():
        tam, _, val = learning_curve(clone(m), X, y, cv=cv, scoring="accuracy",
                                     train_sizes=np.linspace(0.2, 1.0, 6),
                                     n_jobs=-1, random_state=config.SEMILLA)
        ax.plot(tam, val.mean(axis=1), marker="o", label=nombre)
        ax.fill_between(tam, val.mean(axis=1) - val.std(axis=1),
                        val.mean(axis=1) + val.std(axis=1), alpha=0.15)
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("Exactitud (validación cruzada)")
    ax.set_title("Curvas de aprendizaje")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "curvas_aprendizaje.png", dpi=150)
    plt.close(fig)
