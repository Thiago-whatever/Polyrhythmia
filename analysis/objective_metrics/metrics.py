from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class ObjectiveMetrics:
    density: np.ndarray   # (N,)
    entropy: np.ndarray   # (N,)
    inst_counts: np.ndarray  # (N, I)


def rhythmic_density(M: np.ndarray) -> np.ndarray:
    """
    Densidad rítmica por barra.

    Definición (robusta y comparable):
      density = (# hits totales en la barra) / (T * I)

    Donde M es (N,T,I) binaria.
    Resultado en [0,1].
    """
    N, T, I = M.shape
    hits = M.sum(axis=(1, 2))
    return hits / float(T * I)


def token_entropy_from_bitmasks(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Entropía rítmica por barra basada en distribución de 'tokens' implícitos:
    Cada paso t es un vector binario de instrumentos. Consideramos cada vector
    como un "símbolo" y calculamos la entropía de la distribución de símbolos
    dentro de la barra.

    Procedimiento por barra:
      - Para cada paso t, convertimos el vector binario a una tupla (token).
      - Estimamos p(token) dentro de la barra (frecuencias / T).
      - H = - sum p log2 p

    Interpretación:
      - H baja: patrón repetitivo (muchos pasos iguales).
      - H alta: mayor variación a lo largo del compás.

    Nota: esta métrica NO es cross-entropy del entrenamiento.
    """
    N, T, I = M.shape
    H = np.zeros((N,), dtype=np.float64)

    for n in range(N):
        # Representación compacta por paso
        rows = [tuple(M[n, t].astype(np.int8).tolist()) for t in range(T)]
        # Conteos
        uniq, counts = np.unique(rows, return_counts=True, axis=0)
        p = counts.astype(np.float64) / float(T)
        p = np.clip(p, eps, 1.0)
        H[n] = -np.sum(p * np.log2(p))
    return H.astype(np.float32)


def instrument_counts(M: np.ndarray) -> np.ndarray:
    """
    Conteo por instrumento por barra: (N,I).
    inst_counts[n,i] = # de pasos en los que el instrumento i está activo en la barra n.
    """
    return M.sum(axis=1).astype(np.float32)


def compute_objective_metrics(M: np.ndarray) -> ObjectiveMetrics:
    return ObjectiveMetrics(
        density=rhythmic_density(M),
        entropy=token_entropy_from_bitmasks(M),
        inst_counts=instrument_counts(M),
    )


def normalize_inst_distribution(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convierte conteos (K,I) a distribución por fila (K,I) sum=1.
    """
    denom = counts.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return counts / denom