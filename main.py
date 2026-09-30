"""Main entry point for the SDSS genetic algorithms pipeline with MongoDB Atlas integration."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pandas as pd

from src.config import FEATURE_COLUMNS, RANDOM_STATE
from src.data_loader import load_dataset, summarize_dataset
from src.feature_selection.ga import run_feature_selection_ga
from src.hyperparameters.ga import run_hyperparameter_ga
from src.clustering.ga import run_clustering_ga


def print_banner() -> None:
    """Print a banner for the genetic algorithms project."""
    print("=" * 55)
    print("ALGORITMOS GENÉTICOS - SDSS (+ MongoDB Atlas)")
    print("=" * 55)


def get_convergence_from_csv(path: Path) -> list[float]:
    """Read the best_fitness column from a convergence.csv file if it exists."""
    if not path.exists():
        return []
    try:
        conv_df = pd.read_csv(path)
        if "best_fitness" in conv_df.columns:
            return [float(x) for x in conv_df["best_fitness"].tolist()]
    except Exception:
        pass
    return []


def main() -> None:
    """Run the dataset validation, feature selection, hyperparameter tuning, and clustering pipeline."""
    print_banner()

    # Inicializar conexión a MongoDB Atlas
    mongo_store = None
    try:
        from src.database.mongodb import MongoRunStore
        mongo_store = MongoRunStore()
        mongo_store.ping()
        mongo_store.ensure_indexes()
        print(f"[MongoDB Atlas] Conexión establecida con éxito:")
        print(f"                Base de datos: {mongo_store.db_name}")
        print(f"                Colección: {mongo_store.collection_name}")
        print(f"                URI: {mongo_store.safe_uri()}\n")
    except Exception as exc:
        print(f"[MongoDB Atlas] Aviso: No se conectó a MongoDB ({exc}). Solo se guardará en outputs/.\n")

    dataset_path = Path(__file__).resolve().parent / "data" / "sdss_sample.csv"
    print("[1/4] Validación del dataset")
    df = load_dataset(dataset_path)
    summary = summarize_dataset(df)
    print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
    print(f"Classes: {summary['class_counts']}")

    # 1. Selección de características
    print("\n[2/4] Selección de características (KNN)")
    t0 = time.time()
    feature_metrics = run_feature_selection_ga(df, output_dir="outputs/feature_selection")
    t_feat = time.time() - t0
    print(f"Mejores variables: {feature_metrics['best_features']}")
    print(f"Accuracy: {feature_metrics['accuracy']:.4f}")

    if mongo_store:
        try:
            conv = get_convergence_from_csv(Path("outputs/feature_selection/convergence.csv"))
            doc_id = mongo_store.log_genetic_run(
                algorithm="genetic_feature_selection",
                problem="astronomy_classification",
                dataset="sdss_sample.csv",
                seed=RANDOM_STATE,
                params={
                    "population_size": feature_metrics.get("population_size"),
                    "generations": feature_metrics.get("generations"),
                    "mutation_rate": feature_metrics.get("mutation_rate"),
                    "crossover_rate": feature_metrics.get("crossover_rate"),
                    "best_features": feature_metrics.get("best_features"),
                },
                metrics={
                    "accuracy": feature_metrics.get("accuracy"),
                    "rmse": 1.0 - feature_metrics.get("accuracy", 0.0),
                },
                evaluations=feature_metrics.get("population_size", 20) * feature_metrics.get("generations", 20),
                convergence=conv,
                time_s=t_feat,
                experiment_id="sdss_pipeline_run",
            )
            print(f"[MongoDB Atlas] Guardado en Atlas -> ID: {doc_id}")
        except Exception as exc:
            print(f"[MongoDB Atlas] Error al guardar selección de características: {exc}")

    # 2. Optimización de hiperparámetros
    print("\n[3/4] Optimización de hiperparámetros (Ridge Alpha)")
    t0 = time.time()
    hp_metrics = run_hyperparameter_ga(df, output_dir="outputs/hyperparameters")
    t_hp = time.time() - t0
    print(f"Mejor alpha: {hp_metrics['best_alpha']:.6f}")
    print(f"MSE: {hp_metrics['mse']:.6f}")
    print(f"R²: {hp_metrics['r2']:.6f}")

    if mongo_store:
        try:
            conv = get_convergence_from_csv(Path("outputs/hyperparameters/convergence.csv"))
            doc_id = mongo_store.log_genetic_run(
                algorithm="genetic_ridge_alpha",
                problem="redshift_regression",
                dataset="sdss_sample.csv",
                seed=RANDOM_STATE,
                params={
                    "population_size": hp_metrics.get("population_size"),
                    "generations": hp_metrics.get("generations"),
                    "mutation_rate": hp_metrics.get("mutation_rate"),
                    "best_alpha": hp_metrics.get("best_alpha"),
                },
                metrics={
                    "best_alpha": hp_metrics.get("best_alpha"),
                    "mse": hp_metrics.get("mse"),
                    "r2": hp_metrics.get("r2"),
                    "rmse": float(hp_metrics.get("mse", 0.0) ** 0.5),
                },
                evaluations=hp_metrics.get("population_size", 20) * hp_metrics.get("generations", 20),
                convergence=conv,
                time_s=t_hp,
                experiment_id="sdss_pipeline_run",
            )
            print(f"[MongoDB Atlas] Guardado en Atlas -> ID: {doc_id}")
        except Exception as exc:
            print(f"[MongoDB Atlas] Error al guardar hiperparámetros: {exc}")

    # 3. Clustering evolutivo
    print("\n[4/4] Clustering evolutivo (Centroides dinámicos)")
    t0 = time.time()
    cluster_metrics = run_clustering_ga(df, output_dir="outputs/clustering")
    t_clust = time.time() - t0
    print(f"SSE AG: {cluster_metrics['genetic_sse']:.6f}")
    print(f"SSE KMeans: {cluster_metrics['kmeans_sse']:.6f}")

    if mongo_store:
        try:
            conv = get_convergence_from_csv(Path("outputs/clustering/convergence.csv"))
            doc_id = mongo_store.log_genetic_run(
                algorithm="genetic_clustering",
                problem="sdss_clustering",
                dataset="sdss_sample.csv",
                seed=RANDOM_STATE,
                params={
                    "population_size": cluster_metrics.get("population_size"),
                    "generations": cluster_metrics.get("generations"),
                    "mutation_rate": cluster_metrics.get("mutation_rate"),
                    "k_clusters": 3,
                },
                metrics={
                    "genetic_sse": cluster_metrics.get("genetic_sse"),
                    "kmeans_sse": cluster_metrics.get("kmeans_sse"),
                    "rmse": float(cluster_metrics.get("genetic_sse", 0.0)),
                },
                evaluations=cluster_metrics.get("population_size", 20) * cluster_metrics.get("generations", 20),
                convergence=conv,
                time_s=t_clust,
                experiment_id="sdss_pipeline_run",
            )
            print(f"[MongoDB Atlas] Guardado en Atlas -> ID: {doc_id}")
        except Exception as exc:
            print(f"[MongoDB Atlas] Error al guardar clustering: {exc}")

    summary_payload = {
        "feature_selection": feature_metrics,
        "hyperparameters": hp_metrics,
        "clustering": cluster_metrics,
    }
    Path("outputs").mkdir(exist_ok=True)
    (Path("outputs") / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\nProceso finalizado exitosamente.")
    print("Resultados guardados en outputs/ y registrados en tu clúster de MongoDB Atlas.")


if __name__ == "__main__":
    main()

