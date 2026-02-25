"""
stats_chi2.py
-------------
Prueba chi-cuadrado (χ²) de independencia para:
  Origen real (Humano vs IA)  X  Respuesta (HUMANO / IA / NO_SEGURO)

Se usa cuando los conteos esperados no son pequeños. En tu caso,
para una tabla 2×3 con ~440 observaciones, χ² es apropiada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


@dataclass(frozen=True)
class Chi2Diagnostics:
    min_expected: float
    num_expected_lt_5: int
    num_expected_lt_1: int
    expected_shape: Tuple[int, int]


@dataclass(frozen=True)
class Chi2Result:
    chi2: float
    dof: int
    p_value: float
    contingency_table: pd.DataFrame
    expected: np.ndarray
    diagnostics: Chi2Diagnostics


def chi2_independence_test(
    df_long: pd.DataFrame,
    true_col: str = "true_origin",
    resp_col: str = "response_norm",
    order_true: List[str] | None = None,
    order_resp: List[str] | None = None,
) -> Chi2Result:
    """
    Ejecuta χ² de independencia.

    df_long debe tener columnas:
      - true_origin: "Humano" / "IA" (o etiquetas equivalentes)
      - response_norm: "HUMANO" / "IA" / "NO_SEGURO"

    order_true y order_resp se usan para forzar orden en la tabla.
    """
    if order_true is None:
        order_true = ["Humano", "IA"]
    if order_resp is None:
        order_resp = ["HUMANO", "IA", "NO_SEGURO"]

    # Tabla de contingencia
    ct = pd.crosstab(df_long[true_col], df_long[resp_col])

    # Reindex para asegurar orden consistente (y llenar faltantes con 0)
    ct = ct.reindex(index=order_true, columns=order_resp, fill_value=0)

    chi2, p, dof, expected = chi2_contingency(ct.values)

    exp = np.asarray(expected, dtype=float)
    diag = Chi2Diagnostics(
        min_expected=float(exp.min()),
        num_expected_lt_5=int((exp < 5).sum()),
        num_expected_lt_1=int((exp < 1).sum()),
        expected_shape=exp.shape,
    )

    return Chi2Result(
        chi2=float(chi2),
        dof=int(dof),
        p_value=float(p),
        contingency_table=ct,
        expected=exp,
        diagnostics=diag,
    )