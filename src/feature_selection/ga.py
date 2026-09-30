"""Feature selection genetic algorithm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CROSSOVER_RATE, ELITISM, GENERATIONS, MUTATION_RATE, POPULATION_SIZE, RANDOM_STATE, TARGET_COLUMN
from src.feature_selection.chromosome import FEATURE_COLUMNS, random_feature_chromosome, repair_feature_chromosome, validate_feature_chromosome
from src.feature_selection.fitness import feature_selection_fitness
from src.genetic.crossover import one_point_crossover
from src.genetic.mutation import bit_flip_mutation
from src.genetic.selection import tournament_selection
from src.genetic.utils import evaluate_diversity
from src.visualization import save_convergence_plot, save_confusion_matrix_plot


def run_feature_selection_ga(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/feature_selection",
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    generations: int = GENERATIONS,
) -> dict:
    """Run the feature-selection GA and save metrics and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = df[list(FEATURE_COLUMNS)].copy()
    target = df[TARGET_COLUMN].copy()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=RANDOM_STATE, stratify=target)

    X_train_np = X_train.to_numpy(dtype=float)
    X_test_np = X_test.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    population = np.array([random_feature_chromosome() for _ in range(population_size)], dtype=int)
    history = []
    best_chromosome = None
    best_fitness = -np.inf

    for generation in range(1, generations + 1):
        fitness = np.array([
            feature_selection_fitness(chromosome, X_train_np, X_test_np, y_train_np, y_test_np)
            for chromosome in population
        ])
        idx_best = int(np.argmax(fitness))
        generation_best = float(fitness[idx_best])
        generation_average = float(np.mean(fitness))
        diversity = evaluate_diversity(population)
        history.append({
            "generation": generation,
            "best_fitness": generation_best,
            "average_fitness": generation_average,
            "population_diversity": diversity,
        })

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_chromosome = population[idx_best].copy()

        new_population = []
        if ELITISM > 0:
            elites = fitness.argsort()[-ELITISM:]
            for idx in elites:
                new_population.append(population[idx].copy())

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            child1 = repair_feature_chromosome(child1)
            child2 = repair_feature_chromosome(child2)
            if np.random.random() < mutation_rate:
                child1 = bit_flip_mutation(child1, mutation_rate=mutation_rate)
                child1 = repair_feature_chromosome(child1)
            if np.random.random() < mutation_rate:
                child2 = bit_flip_mutation(child2, mutation_rate=mutation_rate)
                child2 = repair_feature_chromosome(child2)
            new_population.extend([child1, child2])
        population = np.array(new_population[:population_size], dtype=int)

    final_best = best_chromosome if best_chromosome is not None else population[0]
    selected_idx = np.where(final_best == 1)[0]
    features_selected = [FEATURE_COLUMNS[index] for index in selected_idx]
    model = __import__("sklearn.neighbors", fromlist=["KNeighborsClassifier"]).KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_np[:, selected_idx], y_train_np)
    predictions = model.predict(X_test_np[:, selected_idx])
    accuracy = float(np.mean(predictions == y_test_np))

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "convergence.csv", index=False)
    save_convergence_plot(history_df, output_dir / "convergence.png", "Feature Selection GA Convergence")
    save_confusion_matrix_plot(y_test_np, predictions, output_dir / "confusion_matrix.png")

    metrics = {
        "best_features": features_selected,
        "best_chromosome": final_best.tolist(),
        "accuracy": accuracy,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": CROSSOVER_RATE,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
