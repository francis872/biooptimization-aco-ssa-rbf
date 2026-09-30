import numpy as np

from src.clustering.chromosome import random_centroid_chromosome


def test_clustering_chromosome_length_is_15():
    matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0, 7.0]])
    chromosome = random_centroid_chromosome(matrix)
    assert chromosome.shape == (15,)
    assert np.isfinite(chromosome).all()
