"""Evolutionary clustering GA using centroid chromosomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.chromosome import clip_centroids, random_centroid_chromosome, validate_clustering_chromosome
from src.clustering.fitness import clustering_fitness, clustering_sse
from src.config import CROSSOVER_RATE, ELITISM, GENERATIONS, MUTATION_RATE, POPULATION_SIZE, RANDOM_STATE
from src.genetic.crossover import arithmetic_crossover
from src.genetic.mutation import gaussian_mutation
from src.genetic.selection import tournament_selection
from src.genetic.utils import evaluate_diversity
from src.visualization import plot_cluster_results, plot_real_classes, save_convergence_plot


def run_clustering_ga(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/clustering",
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    generations: int = GENERATIONS,
) -> dict:
    """Run the centroid-evolution GA and compare against KMeans."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[["u", "g", "r", "i", "z"]].to_numpy(dtype=float)
    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)

    population = np.array([random_centroid_chromosome(X) for _ in range(population_size)], dtype=float)
    population = np.array([clip_centroids(chromosome, feature_min, feature_max) for chromosome in population], dtype=float)
    history = []
    best_chromosome = None
    best_fitness = -np.inf
    best_sse = np.inf

    for generation in range(1, generations + 1):
        fitness_values = np.array([clustering_fitness(X, chromosome, num_centroids=3) for chromosome in population])
        idx_best = int(np.argmax(fitness_values))
        generation_best = float(fitness_values[idx_best])
        generation_average = float(np.mean(fitness_values))
        diversity = evaluate_diversity(population.reshape(len(population), -1))
        if generation_best > best_fitness:
            best_fitness = generation_best
            best_chromosome = population[idx_best].copy()
            best_sse = clustering_sse(X, best_chromosome)
        history.append({
            "generation": generation,
            "best_fitness": generation_best,
            "average_fitness": generation_average,
            "population_diversity": diversity,
        })

        new_population = []
        if ELITISM > 0:
            elites = np.argsort(fitness_values)[-ELITISM:]
            for idx in elites:
                new_population.append(population[idx].copy())

        while len(new_population) < population_size:
            parent1 = tournament_selection(population.reshape(len(population), -1), fitness_values)
            parent2 = tournament_selection(population.reshape(len(population), -1), fitness_values)
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = arithmetic_crossover(parent1.reshape(-1), parent2.reshape(-1))
            else:
                child1, child2 = parent1.reshape(-1).copy(), parent2.reshape(-1).copy()
            child1 = clip_centroids(child1, feature_min, feature_max)
            child2 = clip_centroids(child2, feature_min, feature_max)
            if np.random.random() < mutation_rate:
                child1 = gaussian_mutation(child1, mutation_rate=mutation_rate, sigma=0.2)
                child1 = clip_centroids(child1, feature_min, feature_max)
            if np.random.random() < mutation_rate:
                child2 = gaussian_mutation(child2, mutation_rate=mutation_rate, sigma=0.2)
                child2 = clip_centroids(child2, feature_min, feature_max)
            new_population.extend([child1, child2])
        population = np.array(new_population[:population_size], dtype=float)

    if best_chromosome is None:
        best_chromosome = population[0].copy()
    genetic_labels = np.array([np.argmin(np.linalg.norm(X[:, None, :] - best_chromosome.reshape(3, 5)[None, :, :], axis=2), axis=1)])
    genetic_labels = genetic_labels.reshape(-1)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "convergence.csv", index=False)
    save_convergence_plot(history_df, output_dir / "convergence.png", "Evolutionary Clustering Convergence")

    plot_cluster_results(X, genetic_labels, "Evolutionary Clustering", output_dir / "genetic_clusters.png")
    plot_cluster_results(X, kmeans_labels, "KMeans Clustering", output_dir / "kmeans_clusters.png")
    plot_real_classes(X, df["class"].to_numpy(), output_dir / "real_classes.png")

    genetic_sse = clustering_sse(X, best_chromosome)
    kmeans_sse = float(np.sum((X - kmeans.cluster_centers_[kmeans_labels]) ** 2))
    metrics = {
        "genetic_sse": genetic_sse,
        "kmeans_sse": kmeans_sse,
        "best_fitness": best_fitness,
        "population_size": population_size,
        "generations": generations,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
