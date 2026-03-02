from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np


@dataclass
class BitmaskDataset:
    """Dataset de barras en forma (N, T, I) binaria + metadatos opcionales."""
    M: np.ndarray  # (N,T,I) float32 o int8
    genres: Optional[np.ndarray] = None  # (N, G) multi-hot o (N,) labels
    meta: Optional[Dict[str, Any]] = None


def _strings_row_to_matrix(strings_row: np.ndarray) -> np.ndarray:
    """
    Convierte una fila de T strings (cada string '01001...') a (T,I) int8.
    """
    T = len(strings_row)
    I = len(str(strings_row[0]))
    M = np.zeros((T, I), dtype=np.int8)
    for t, s in enumerate(strings_row):
        s = str(s)
        # robusto por si viene como bytes
        if s.startswith("b'") and s.endswith("'"):
            s = s[2:-1]
        for i, ch in enumerate(s):
            if ch != "0":
                M[t, i] = 1
    return M


def _find_first_string_array(npz: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    """
    Busca en un npz el primer arreglo que parezca contener strings bitmask.
    Espera forma (N, T) con elementos tipo str/bytes.
    """
    for k, arr in npz.items():
        if arr.dtype.kind in ("U", "S", "O") and arr.ndim == 2:
            # Heurística: elementos parecen 0/1
            sample = arr.flat[0]
            s = str(sample)
            if "0" in s and "1" in s:
                return k, arr
    raise ValueError("No se encontró ningún array (N,T) de strings bitmask en el npz.")


def _find_genre_array(npz: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Intenta hallar género/estilo: keys típicos contienen 'genre' o 'style'.
    Puede ser multi-hot (N,G) o labels (N,).
    """
    candidates = []
    for k, arr in npz.items():
        lk = k.lower()
        if "genre" in lk or "style" in lk:
            candidates.append(arr)
    if not candidates:
        return None
    # Preferir multi-hot 2D
    for arr in candidates:
        if arr.ndim == 2:
            return arr
    return candidates[0]


def load_bitmasks_npz(path: str | Path) -> BitmaskDataset:
    """
    Carga un *.npz con bitmasks tipo strings (N,T) y lo convierte a (N,T,I).
    Compatible con tu formato de train_bitmasks/test_bitmasks usado en notebooks.
    """
    path = Path(path)
    npz = np.load(path, allow_pickle=True)

    key_masks, masks = _find_first_string_array(npz)
    N, T = masks.shape
    # Convertir todas las barras
    M = np.zeros((N, T, len(str(masks[0, 0]))), dtype=np.int8)
    for n in range(N):
        M[n] = _strings_row_to_matrix(masks[n])

    genres = _find_genre_array(npz)

    meta = {"source_path": str(path), "key_masks": key_masks, "keys": list(npz.keys())}
    return BitmaskDataset(M=M.astype(np.float32), genres=genres, meta=meta)