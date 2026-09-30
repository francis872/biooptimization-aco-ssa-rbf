# Bio-Optimization: ACO · SSA · RBF con MongoDB Atlas

Proyecto académico para aplicar ACO, SSA y una red RBF optimizada con SSA, almacenando las corridas experimentales en MongoDB.

## Funcionalidades

- ACO para TSP con al menos 15 ciudades y comparación con heurística.
- SSA para minimizar Rastrigin y curva de convergencia.
- RBF base frente a RBF optimizada con SSA.
- Experimento principal con 10 semillas.
- Registro de parámetros, métricas, tiempo, evaluaciones y convergencia en MongoDB.
- Interfaz web HTML/CSS/JavaScript servida por Python.
- Pruebas automáticas.
- Notebook de análisis.
- Dockerfile y Jenkinsfile.

## Instalación

```bash
pip install -r requirements.txt
```

## MongoDB Atlas

Copia `.env.example` como `.env` y configura:

```env
MONGO_URI=mongodb+srv://USUARIO:CONTRASENA@CLUSTER/?retryWrites=true&w=majority
MONGO_DB=biooptimization
MONGO_COLLECTION=runs
```

El archivo `.env` está ignorado por Git y no debe subirse.

## Interfaz web

```bash
python web_app.py
```

Abre `http://127.0.0.1:5000`.

La interfaz permite ejecutar ACO/TSP, SSA/Rastrigin y RBF vs. SSA-RBF, visualizar convergencia y consultar resultados de MongoDB.

## Pipeline completo

```bash
python main.py
```

## Pruebas

```bash
pytest tests/ -v
```

## MongoDB: consultas para sustentación

```python
from src.database.mongodb import MongoRunStore
store = MongoRunStore()

store.best_run("ssa_rbf", metric="rmse")

list(store.collection.aggregate([
    {"$match": {"algorithm": "ssa_rbf"}},
    {"$group": {"_id": None, "avg_rmse": {"$avg": "$metrics.rmse"}}}
]))

store.aggregate_by_algorithm()
```

## Estructura principal

```text
src/
  aco/
  ssa/
  rbf/
  database/
tests/
notebooks/
results/
templates/
static/
main.py
web_app.py
requirements.txt
README.md
```
