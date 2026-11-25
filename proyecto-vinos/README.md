# Clasificación de vinos por composición química: Random Forest vs KNN

Proyecto de curso – Introducción a la Inteligencia Artificial (2026–2027).

**Pregunta experimental:** ¿qué algoritmo clasifica mejor el origen de un vino
(3 clases) a partir de 13 atributos químicos: Random Forest o KNN, ambos con
ajuste riguroso de hiperparámetros?

## Estructura

```
proyecto-vinos/
├── run_experiment.py      # Ejecuta TODO el experimento (un solo comando)
├── requirements.txt
├── src/
│   ├── config.py          # Semilla, partición, rejillas de hiperparámetros
│   ├── datos.py           # Carga del dataset, partición 80/20 y EDA
│   ├── modelos.py         # Pipelines (StandardScaler + clasificador)
│   └── experimento.py     # Tuning, evaluación, significancia, importancias
├── tests/test_pipeline.py # Pruebas mínimas del espacio de trabajo
├── resultados/            # Se genera al ejecutar (CSV, JSON, TXT, figuras)
└── informe/informe.md     # Esqueleto del informe con los resultados obtenidos
```

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiment.py        # ~1 minuto en un portátil
python -m pytest -q             # opcional: verifica el entorno
```

Todo es determinista (`SEMILLA = 42` en `src/config.py`). Los datos vienen
incluidos en scikit-learn (`sklearn.datasets.load_wine`); no hay que descargar nada.

## Diseño del experimento

1. **Datos:** Wine dataset (178 muestras, 13 atributos numéricos, 3 clases).
2. **Partición:** 80/20 estratificada (142 entrenamiento / 36 prueba).
3. **Preproceso:** `StandardScaler` dentro de un `Pipeline` (evita fuga de datos).
4. **Ajuste:** `GridSearchCV`, 5-fold estratificado, métrica F1 macro.
   - RF: 64 combinaciones (árboles, profundidad, `min_samples_split`, `max_features`).
   - KNN: 168 combinaciones (k, métrica, pesos, estructura de índice).
5. **Evaluación final:** exactitud, F1 macro, matriz de confusión y tiempos.
6. **Significancia:** 10×(5-fold) sobre todo el dataset, t-test pareado con
   corrección de Nadeau–Bengio y Wilcoxon.
7. **Extensiones incluidas:** importancia de atributos (Gini y permutación) y
   curvas de aprendizaje.

## Salidas principales (`resultados/`)

| Archivo | Contenido |
|---|---|
| `comparacion.csv` | Métricas de ambos modelos en el conjunto de prueba |
| `tuning.json` | Mejores hiperparámetros, F1 en CV y tiempo de ajuste |
| `significancia.json` | Resultados de las pruebas estadísticas |
| `importancia_rasgos.csv` | Importancia de cada atributo |
| `reporte_clasificacion.txt` | Precisión/recall/F1 por clase |
| `figuras/*.png` | EDA, matrices de confusión, importancias, curvas de aprendizaje |
