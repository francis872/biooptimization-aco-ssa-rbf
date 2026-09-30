import numpy as np

from src.genetic.crossover import arithmetic_crossover, one_point_crossover
from src.genetic.mutation import bit_flip_mutation, gaussian_mutation
from src.genetic.selection import tournament_selection


def test_tournament_selection_returns_valid_individual():
    population = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=int)
    fitness = np.array([0.2, 0.7, 0.9])
    winner = tournament_selection(population, fitness)
    assert winner.shape == (3,)
    assert set(np.unique(winner)).issubset({0, 1})


def test_one_point_crossover_keeps_length():
    parent1 = np.array([1, 0, 1, 0, 1])
    parent2 = np.array([0, 1, 1, 1, 0])
    child1, child2 = one_point_crossover(parent1, parent2)
    assert child1.shape == parent1.shape
    assert child2.shape == parent2.shape


def test_bit_flip_mutation_preserves_binary_genes():
    chromosome = np.array([1, 0, 1, 0, 1], dtype=int)
    mutated = bit_flip_mutation(chromosome, mutation_rate=1.0)
    assert set(np.unique(mutated)).issubset({0, 1})


def test_gaussian_mutation_keeps_dimensions():
    chromosome = np.array([0.2, 0.4, 0.6], dtype=float)
    mutated = gaussian_mutation(chromosome, mutation_rate=1.0, sigma=0.5)
    assert mutated.shape == chromosome.shape
