"""
cramers_v.py
------------
Cálculo de Cramér's V como tamaño de efecto para una tabla de contingencia RxC.

Definición:
  V = sqrt( chi2 / (n * (k - 1)) )

donde:
  - chi2: estadístico chi-cuadrado de independencia
  - n: total de observaciones
  - k: min(#filas, #columnas)

Para tu caso (2x3), k = 2  => (k - 1) = 1
Entonces V = sqrt(chi2 / n)

Nota: si aplicas corrección de Yates, eso es para 2x2 y no aplica aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CramersVResult:
    v: float
    n: int
    k: int
    table_shape: Tuple[int, int]


def cramers_v_from_table(contingency_table: pd.DataFrame | np.ndarray, chi2: float) -> CramersVResult:
    """
    Calcula Cramér's V a partir de:
      - contingency_table (DataFrame o ndarray)
      - chi2 (estadístico ya calculado)

    Devuelve: V, n, k, shape
    """
    if isinstance(contingency_table, pd.DataFrame):
        table = contingency_table.values
        shape = contingency_table.shape
    else:
        table = np.asarray(contingency_table)
        shape = table.shape

    n = int(table.sum())
    if n <= 0:
        raise ValueError("La tabla de contingencia tiene n=0; no se puede calcular Cramér's V.")

    r, c = shape
    k = int(min(r, c))
    if k <= 1:
        raise ValueError("Cramér's V requiere al menos 2 filas y 2 columnas.")

    denom = n * (k - 1)
    v = float(np.sqrt(chi2 / denom))
    return CramersVResult(v=v, n=n, k=k, table_shape=shape)