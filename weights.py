"""Вычисление весов факторов из ранговых отношений RASP."""
from __future__ import annotations
import numpy as np


def compute_coefficients(q):
    """
    Сырые коэффициенты значимости c_j из отношений q_j = c_j / c_{j+1}.
    c_n = 1, c_j = prod_{s=j}^{n-1} q_s.

    q : array-like, длина n-1, все q_j >= 1.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q должен быть одномерным массивом")
    if q.size == 0:
        return np.array([1.0])
    if np.any(q < 1.0):
        raise ValueError(f"Все q_j должны быть >= 1, получено: {q.tolist()}")
    return np.concatenate([np.cumprod(q[::-1])[::-1], [1.0]])


def compute_weights(q):
    """
    Нормированные веса факторов: w_j = c_j / sum_k c_k,  sum_j w_j = 1.
    """
    c = compute_coefficients(q)
    return c / c.sum()