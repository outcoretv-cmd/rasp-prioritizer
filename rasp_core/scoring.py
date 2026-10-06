"""Агрегация баллов по вероятностям и весам."""
from __future__ import annotations
import numpy as np


def compute_scores(P, w):
    """S_i = sum_j w_j * p_{i,j}.  P: (M, n), w: (n,)  ->  S: (M,)"""
    P = np.asarray(P, dtype=float)
    w = np.asarray(w, dtype=float)
    if P.shape[1] != w.shape[0]:
        raise ValueError(
            f"Число столбцов P ({P.shape[1]}) не совпадает "
            f"с числом весов ({w.shape[0]})"
        )
    return P @ w


def total_scores(R, O, lam):
    """V_i = λ R_i + (1 − λ) O_i."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"λ должен быть в [0, 1], получено: {lam}")
    return lam * np.asarray(R) + (1.0 - lam) * np.asarray(O)