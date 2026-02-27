"""
wilcoxon_neutral.py
-------------------
Wilcoxon signed-rank test contra valor neutro (por defecto 3).

H0: mediana(x - neutral) = 0
H1: mediana(x - neutral) != 0 (two-sided por defecto)

Nota: Wilcoxon ignora ceros (x==neutral) en SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_total: int
    n_used: int
    neutral: float


def wilcoxon_vs_neutral(x: pd.Series, neutral: float = 3.0, alternative: str = "two-sided") -> WilcoxonResult:
    vals = pd.to_numeric(x, errors="coerce").dropna().astype(float).values
    n_total = int(len(vals))
    if n_total == 0:
        raise ValueError("No hay datos numéricos válidos para Wilcoxon.")

    diffs = vals - neutral
    # SciPy wilcoxon ignora ceros; n_used = count(diffs != 0)
    n_used = int((diffs != 0).sum())
    if n_used == 0:
        # todos son exactamente neutral -> p=1, stat=0 de facto
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_total=n_total, n_used=0, neutral=neutral)

    res = wilcoxon(diffs, alternative=alternative, zero_method="wilcox", correction=False, method="auto")
    return WilcoxonResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_total=n_total,
        n_used=n_used,
        neutral=float(neutral),
    )