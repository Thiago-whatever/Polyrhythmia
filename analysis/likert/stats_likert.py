"""
stats_likert.py
---------------
- Resúmenes por grupo: mediana e IQR (Q3-Q1)
- Mann–Whitney U: compara IA vs Humano (no paramétrico)
- (Opcional) Kruskal–Wallis: compara distribución de ratings entre audios (1..4)

Nota metodológica:
Mann–Whitney asume muestras independientes. Aquí cada participante califica múltiples audios,
por lo que hay dependencia intra-participante. Aun así, muchos trabajos reportan MWU como
aproximación simple. Si luego quieres una versión "paired", se puede hacer Wilcoxon
sobre promedios IA vs Humano por participante (lo vemos si lo piden).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal


@dataclass(frozen=True)
class MedianIQR:
    n: int
    median: float
    q1: float
    q3: float
    iqr: float


def median_iqr(x: pd.Series) -> MedianIQR:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return MedianIQR(n=0, median=float("nan"), q1=float("nan"), q3=float("nan"), iqr=float("nan"))
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    med = float(x.median())
    return MedianIQR(n=int(len(x)), median=med, q1=q1, q3=q3, iqr=q3 - q1)


def attach_truth_origin(df_long: pd.DataFrame, truth_map: Dict[int, str]) -> pd.DataFrame:
    """
    Agrega columna true_origin ("IA" / "Humano") usando audio_index.
    """
    df = df_long.copy()
    df["true_origin"] = df["audio_index"].map(truth_map)
    if df["true_origin"].isna().any():
        missing = df[df["true_origin"].isna()]["audio_index"].unique().tolist()
        raise ValueError(f"Faltan audio_index en truth_map: {missing}")
    return df


@dataclass(frozen=True)
class MannWhitneyResult:
    u: float
    p_value_two_sided: float
    n_ia: int
    n_humano: int


def mann_whitney_ia_vs_humano(df_with_truth: pd.DataFrame) -> MannWhitneyResult:
    """
    Mann–Whitney U (two-sided) comparando ratings de audios IA vs Humano.
    """
    ia = df_with_truth[df_with_truth["true_origin"] == "IA"]["rating"].astype(float)
    hu = df_with_truth[df_with_truth["true_origin"] == "Humano"]["rating"].astype(float)

    # scipy recomienda method="auto" (elige exact/asy según tamaños)
    res = mannwhitneyu(ia, hu, alternative="two-sided", method="auto")

    return MannWhitneyResult(
        u=float(res.statistic),
        p_value_two_sided=float(res.pvalue),
        n_ia=int(len(ia)),
        n_humano=int(len(hu)),
    )


@dataclass(frozen=True)
class KruskalResult:
    h: float
    p_value: float
    groups: List[int]
    n_total: int


def kruskal_by_audio(df_long: pd.DataFrame) -> KruskalResult:
    """
    Kruskal–Wallis para comparar si al menos un audio tiene distribución de ratings distinta.
    Grupos: audio_index 1..4 (los que existan).
    """
    groups = sorted(df_long["audio_index"].unique().tolist())
    samples = [df_long[df_long["audio_index"] == g]["rating"].astype(float).values for g in groups]

    if len(samples) < 2:
        raise ValueError("Kruskal–Wallis requiere al menos 2 grupos (audios).")

    stat = kruskal(*samples)
    return KruskalResult(
        h=float(stat.statistic),
        p_value=float(stat.pvalue),
        groups=[int(g) for g in groups],
        n_total=int(len(df_long)),
    )


def summary_by_origin(df_with_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Mediana e IQR por origen (IA vs Humano).
    """
    rows = []
    for origin in ["IA", "Humano"]:
        s = df_with_truth[df_with_truth["true_origin"] == origin]["rating"]
        m = median_iqr(s)
        rows.append(
            {
                "true_origin": origin,
                "n": m.n,
                "median": m.median,
                "q1": m.q1,
                "q3": m.q3,
                "iqr": m.iqr,
            }
        )
    return pd.DataFrame(rows)


def summary_by_audio(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Mediana e IQR por audio (1..4).
    """
    rows = []
    for audio_idx in sorted(df_long["audio_index"].unique().tolist()):
        s = df_long[df_long["audio_index"] == audio_idx]["rating"]
        m = median_iqr(s)
        rows.append(
            {
                "audio_index": int(audio_idx),
                "n": m.n,
                "median": m.median,
                "q1": m.q1,
                "q3": m.q3,
                "iqr": m.iqr,
            }
        )
    return pd.DataFrame(rows)