"""Hyperparameter optimization genetic algorithm for Ridge alpha."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CROSSOVER_RATE, ELITISM, GENERATIONS, MUTATION_RATE, POPULATION_SIZE, RANDOM_STATE
from src.genetic.crossover import arithmetic_crossover
from src.genetic.mutation import gaussian_mutation
from src.genetic.selection import tournament_selection
from src.genetic.utils import evaluate_diversity
from src.hyperparameters.chromosome import ALPHA_MAX, ALPHA_MIN, clip_alpha, random_alpha_chromosome, validate_alpha_chromosome
from src.hyperparameters.fitness import alpha_fitness
from src.visualization import save_convergence_plot


def run_hyperparameter_ga(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/hyperparameters",
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    generations: int = GENERATIONS,
) -> dict:
    """Optimize alpha for Ridge regression using a real-coded GA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[["u", "g", "r", "i", "z"]].copy()
    y = df["redshift"].copy()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    X_train_np = X_train.to_numpy(dtype=float)
    X_test_np = X_test.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=float)
    y_test_np = y_test.to_numpy(dtype=float)

    population = np.array([random_alpha_chromosome() for _ in range(population_size)], dtype=float)
    history = []
    best_chromosome = None
    best_metrics = {"fitness": -np.inf, "mse": np.inf, "r2": -np.inf}

    for generation in range(1, generations + 1):
        fitness_values = []
        mse_values = []
        r2_values = []
        for chromosome in population:
            alpha = float(chromosome[0])
            fitness, mse, r2 = alpha_fitness(alpha, X_train_np, X_test_np, y_train_np, y_test_np)
            fitness_values.append(fitness)
            mse_values.append(mse)
            r2_values.append(r2)
        fitness_values = np.array(fitness_values)
        mse_values = np.array(mse_values)
        r2_values = np.array(r2_values)

        idx_best = int(np.argmax(fitness_values))
        generation_best = float(fitness_values[idx_best])
        generation_average = float(np.mean(fitness_values))
        diversity = evaluate_diversity(population)
        history.append({
            "generation": generation,
            "best_fitness": generation_best,
            "average_fitness": generation_average,
            "population_diversity": diversity,
        })

        if generation_best > best_metrics["fitness"]:
            best_metrics = {
                "fitness": generation_best,
                "mse": float(mse_values[idx_best]),
                "r2": float(r2_values[idx_best]),
                "alpha": float(population[idx_best][0]),
            }
            best_chromosome = population[idx_best].copy()

        new_population = []
        if ELITISM > 0:
            elites = np.argsort(fitness_values)[-ELITISM:]
            for idx in elites:
                new_population.append(population[idx].copy())

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            child1 = np.array([clip_alpha(float(child1[0]))], dtype=float)
            child2 = np.array([clip_alpha(float(child2[0]))], dtype=float)
            if np.random.random() < mutation_rate:
                child1 = gaussian_mutation(child1, mutation_rate=mutation_rate, sigma=0.5)
                child1 = np.array([clip_alpha(float(child1[0]))], dtype=float)
            if np.random.random() < mutation_rate:
                child2 = gaussian_mutation(child2, mutation_rate=mutation_rate, sigma=0.5)
                child2 = np.array([clip_alpha(float(child2[0]))], dtype=float)
            new_population.extend([child1, child2])
        population = np.array(new_population[:population_size], dtype=float)

    if best_chromosome is None:
        best_chromosome = population[0].copy()
    alpha_best = float(best_chromosome[0])
    final_fitness, final_mse, final_r2 = alpha_fitness(alpha_best, X_train_np, X_test_np, y_train_np, y_test_np)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "convergence.csv", index=False)
    save_convergence_plot(history_df, output_dir / "convergence.png", "Ridge Alpha Optimization Convergence")

    metrics = {
        "best_alpha": alpha_best,
        "mse": final_mse,
        "r2": final_r2,
        "fitness": final_fitness,
        "population_size": population_size,
        "generations": generations,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
