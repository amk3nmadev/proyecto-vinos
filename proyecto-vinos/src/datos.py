"""Carga del Wine dataset y análisis exploratorio (EDA)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from . import config


def cargar_datos():
    """Devuelve (X, y, nombres_de_clase) con X como DataFrame."""
    bunch = load_wine(as_frame=True)
    return bunch.data, bunch.target, list(bunch.target_names)


def particionar(X, y):
    """Partición 80/20 estratificada (la estandarización va dentro del Pipeline
    para evitar fuga de información desde el conjunto de prueba)."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEMILLA
    )


def resumen_eda(X, y, clases):
    """Guarda tablas y figuras de EDA en resultados/."""
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    df = X.copy()
    df["clase"] = [clases[i] for i in y]

    df.drop(columns="clase").describe().T.round(3).to_csv(
        config.DIR_RESULTADOS / "eda_descriptivo.csv")
    df["clase"].value_counts().rename("muestras").to_csv(
        config.DIR_RESULTADOS / "eda_clases.csv")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    sns.countplot(x="clase", data=df, ax=ax, order=clases)
    ax.set_title("Distribución de clases")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "eda_clases.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de correlación de los 13 atributos")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "eda_correlacion.png", dpi=150)
    plt.close(fig)

    sel = ["proline", "flavanoids", "color_intensity", "alcohol"]
    fig, axes = plt.subplots(1, len(sel), figsize=(12, 3.2))
    for ax, col in zip(axes, sel):
        sns.boxplot(x="clase", y=col, data=df, ax=ax, order=clases)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "eda_boxplots.png", dpi=150)
    plt.close(fig)
