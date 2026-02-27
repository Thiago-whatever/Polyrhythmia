"""
io_likert.py
------------
Carga respuestas Likert (1-5) por audio desde Excel y las deja en formato long.

Soporta:
- columnas con sufijos (prefix2, prefix3, etc.)
- columnas duplicadas (pandas las renombra a .1, .2, .3)
- pequeñas variaciones del texto del encabezado (guiones, espacios, etc.)

Estrategia:
- localizar columnas cuyo nombre CONTENGA una "needle" corta (subcadena),
  no el enunciado completo.
- asignar audio_index por orden de aparición.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import re
import unicodedata


# En vez de usar toda la pregunta, usamos una aguja más corta y estable:
LIKERT_NEEDLE = "¿qué tanto te gustó este patrón rítmico"


@dataclass(frozen=True)
class LikertSurveyConfig:
    id_col: str = "ID"
    needle: str = LIKERT_NEEDLE
    expected_num_audios: int = 4


def _normalize_text(s: str) -> str:
    """
    Normaliza para comparar textos:
    - lower
    - quita acentos
    - colapsa espacios
    - reemplaza guiones raros
    """
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_rating(x: object) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = int(float(s))
    except ValueError:
        return None
    if v < 1 or v > 5:
        return None
    return v


def _find_likert_columns_in_order(columns: List[str], needle: str) -> List[str]:
    """
    Devuelve todas las columnas cuyo nombre contenga la needle (normalizada),
    en el orden en que aparecen.
    """
    needle_n = _normalize_text(needle)
    matches: List[str] = []
    for col in columns:
        col_s = _normalize_text(str(col))
        if needle_n in col_s:
            matches.append(col)
    return matches


def load_likert_long(excel_path: str, config: LikertSurveyConfig = LikertSurveyConfig()) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    if config.id_col not in df.columns:
        df = df.copy()
        df[config.id_col] = range(1, len(df) + 1)

    likert_cols = _find_likert_columns_in_order(list(df.columns), config.needle)

    if not likert_cols:
        # Debug útil: muestra columnas para que puedas ver el nombre real
        cols_preview = "\n".join([f"- {c}" for c in df.columns[:50]])
        raise ValueError(
            "No se encontraron columnas Likert.\n"
            f"Needle usada: {config.needle}\n\n"
            "Primeras columnas detectadas en el Excel:\n"
            f"{cols_preview}"
        )

    if len(likert_cols) < config.expected_num_audios:
        raise ValueError(
            f"Se esperaban al menos {config.expected_num_audios} columnas Likert, "
            f"pero se encontraron {len(likert_cols)}.\n"
            f"Columnas Likert encontradas (orden): {likert_cols}"
        )

    # Tomamos las primeras 4 por orden de aparición:
    likert_cols = likert_cols[: config.expected_num_audios]
    audio_map: List[Tuple[int, str]] = [(i + 1, col) for i, col in enumerate(likert_cols)]

    rows = []
    for _, row in df.iterrows():
        pid = row[config.id_col]
        for audio_idx, col in audio_map:
            rating = _parse_rating(row[col])
            if rating is None:
                continue
            rows.append(
                {
                    "participant_id": pid,
                    "audio_index": int(audio_idx),
                    "rating": int(rating),
                }
            )

    return pd.DataFrame(rows)