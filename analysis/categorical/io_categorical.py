"""
io_categorical.py
-----------------
Extrae columnas categóricas por audio desde Excel en formato long.

Soporta encabezados duplicados (pandas: .1, .2, .3) y variaciones leves
buscando por 'needle' (subcadena) normalizada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import unicodedata

import pandas as pd


def _normalize_text(s: str) -> str:
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class CategoricalConfig:
    id_col: str = "ID"
    expected_num_audios: int = 4


def find_columns_by_needle_in_order(columns: List[str], needle: str) -> List[str]:
    needle_n = _normalize_text(needle)
    out: List[str] = []
    for col in columns:
        if needle_n in _normalize_text(str(col)):
            out.append(col)
    return out


def load_categorical_long(
    excel_path: str,
    needle: str,
    config: CategoricalConfig = CategoricalConfig(),
    value_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Retorna long:
      participant_id, audio_index, value_raw, value_norm

    value_map (opcional): mapea valores del Excel a etiquetas canónicas.
    """
    df = pd.read_excel(excel_path)

    if config.id_col not in df.columns:
        df = df.copy()
        df[config.id_col] = range(1, len(df) + 1)

    cols = find_columns_by_needle_in_order(list(df.columns), needle)
    if not cols:
        cols_preview = "\n".join([f"- {c}" for c in df.columns[:60]])
        raise ValueError(
            f"No se encontraron columnas para needle: {needle}\n\n"
            f"Primeras columnas detectadas:\n{cols_preview}"
        )

    if len(cols) < config.expected_num_audios:
        raise ValueError(f"Se esperaban {config.expected_num_audios} columnas, pero se encontraron {len(cols)}: {cols}")

    cols = cols[: config.expected_num_audios]
    audio_map: List[Tuple[int, str]] = [(i + 1, c) for i, c in enumerate(cols)]

    rows = []
    for _, row in df.iterrows():
        pid = row[config.id_col]
        for audio_idx, col in audio_map:
            v = row[col]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            raw = str(v).strip()
            if raw == "":
                continue
            norm = raw
            if value_map is not None:
                norm = value_map.get(raw, f"UNKNOWN::{raw}")
            rows.append(
                {
                    "participant_id": pid,
                    "audio_index": int(audio_idx),
                    "value_raw": raw,
                    "value_norm": norm,
                }
            )

    return pd.DataFrame(rows)