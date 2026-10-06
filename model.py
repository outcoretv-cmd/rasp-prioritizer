"""Ядро метода RASP — высокоуровневый интерфейс."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .scoring import compute_scores, total_scores
from .weights import compute_weights


@dataclass
class RASPResult:
    deal_names: list
    R: np.ndarray
    O: np.ndarray
    V: np.ndarray
    order: list
    lam: float
    w_risk: np.ndarray
    w_opp: np.ndarray

    def ranked_deals(self) -> list:
        return [self.deal_names[i] for i in self.order]

    def summary(self) -> str:
        lines = [f"λ = {self.lam:.2f}", ""]
        for rank, i in enumerate(self.order, start=1):
            lines.append(
                f"{rank}. {self.deal_names[i]:<20s} "
                f"R={self.R[i]:.3f}  O={self.O[i]:.3f}  V={self.V[i]:.3f}"
            )
        return "\n".join(lines)


class RASPMethod:
    """
    Метод приоритизации RASP.

    Пример
    ------
    >>> m = RASPMethod(q_risk=[3, 2], q_opp=[2, 1.5])
    >>> res = m.rank(
    ...     P_risk=[[0.8, 0.5, 0.3], [0.4, 0.9, 0.6], [0.7, 0.2, 0.9]],
    ...     P_opp=[[0.6, 0.7, 0.4], [0.9, 0.3, 0.5], [0.2, 0.8, 0.7]],
    ...     lam=0.6, deal_names=["A", "B", "C"],
    ... )
    >>> res.ranked_deals()
    ['A', 'B', 'C']
    """

    def __init__(self, q_risk: Sequence[float], q_opp: Sequence[float]):
        self.w_risk = compute_weights(q_risk)
        self.w_opp = compute_weights(q_opp)

    def rank(self, P_risk, P_opp, lam=0.5, deal_names=None) -> RASPResult:
        P_risk = np.asarray(P_risk, dtype=float)
        P_opp = np.asarray(P_opp, dtype=float)

        if P_risk.ndim != 2 or P_opp.ndim != 2:
            raise ValueError("P_risk и P_opp должны быть двумерными")
        if P_risk.shape[0] != P_opp.shape[0]:
            raise ValueError("P_risk и P_opp должны иметь одинаковое число строк")
        if P_risk.shape[1] != self.w_risk.size:
            raise ValueError(
                f"P_risk: {P_risk.shape[1]} столбцов, "
                f"а весов рисков {self.w_risk.size}"
            )
        if P_opp.shape[1] != self.w_opp.size:
            raise ValueError(
                f"P_opp: {P_opp.shape[1]} столбцов, "
                f"а весов возможностей {self.w_opp.size}"
            )
        for name, P in (("P_risk", P_risk), ("P_opp", P_opp)):
            if np.any(P < 0) or np.any(P > 1):
                raise ValueError(f"{name}: вероятности должны быть в [0, 1]")

        M = P_risk.shape[0]
        if deal_names is None:
            deal_names = [f"Дело {i+1}" for i in range(M)]
        if len(deal_names) != M:
            raise ValueError("Длина deal_names не совпадает с числом дел")

        R = compute_scores(P_risk, self.w_risk)
        O = compute_scores(P_opp, self.w_opp)
        V = total_scores(R, O, lam)
        order = list(np.argsort(-V, kind="stable"))

        return RASPResult(
            deal_names=list(deal_names), R=R, O=O, V=V, order=order,
            lam=lam, w_risk=self.w_risk, w_opp=self.w_opp,
        )