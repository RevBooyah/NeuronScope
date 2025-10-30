import numpy as np
from src.backend.comparison.cross_model_comparison import CrossModelComparator


def test_cosine_similarity_matrix_basic():
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim = CrossModelComparator._cosine_similarity_matrix(A, B)
    assert sim.shape == (2, 2)
    assert np.allclose(sim, np.eye(2), atol=1e-6)


def test_cosine_similarity_matrix_orthogonal():
    A = np.array([[1.0, 0.0]], dtype=np.float32)
    B = np.array([[0.0, 1.0]], dtype=np.float32)
    sim = CrossModelComparator._cosine_similarity_matrix(A, B)
    assert sim.shape == (1, 1)
    assert abs(sim[0, 0]) < 1e-6


def test_cosine_similarity_matrix_empty():
    A = np.zeros((0, 2), dtype=np.float32)
    B = np.zeros((0, 2), dtype=np.float32)
    sim = CrossModelComparator._cosine_similarity_matrix(A, B)
    assert sim.shape == (0, 0)

