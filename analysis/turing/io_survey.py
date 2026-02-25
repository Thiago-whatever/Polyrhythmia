"""
io_survey.py
------------
Utilidades para cargar el Excel de respuestas y extraer las columnas
de "Prueba de autoría" (Humano / IA / No seguro) en formato long.

Formato long: una fila por (participante, audio), con:
- participant_id
- audio_index (1..4)
- response_raw (texto original)
- response_norm (HUMANO / IA / NO_SEGURO)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


AUTHORSHIP_COL_PREFIX = "Según tu impresión, este patrón fue creado por..."


# Mapeo robusto para normalizar respuestas del Excel a etiquetas canónicas.
# Ajusta aquí si en el futuro hay variantes de texto.
RESPONSE_NORMALIZATION_MAP: Dict[str, str] = {
    "Un humano": "HUMANO",
    "Una Inteligencia Artificial": "IA",
    "No estoy seguro": "NO_SEGURO",
}


@dataclass(frozen=True)
class SurveyConfig:
    """
    Configuración de parsing del Excel.
    """
    id_col: str = "ID"
    authorship_prefix: str = AUTHORSHIP_COL_PREFIX


def _find_authorship_columns(columns: List[str], prefix: str) -> List[Tuple[int, str]]:
    """
    Encuentra columnas de autoría:
      - prefix (audio 1)
      - prefix + "2" (audio 2)
      - prefix + "3" (audio 3)
      - prefix + "4" (audio 4)

    Devuelve lista ordenada: [(1, colname), (2, colname2), ...]
    """
    found: List[Tuple[int, str]] = []

    # Audio 1 exacto
    if prefix in columns:
        found.append((1, prefix))

    # Audios 2..N con sufijo numérico
    # (en tu Excel actual son 2,3,4)
    for i in range(2, 20):
        cand = f"{prefix}{i}"
        if cand in columns:
            found.append((i, cand))

    found.sort(key=lambda x: x[0])
    return found


def _normalize_response(resp: object) -> Optional[str]:
    """
    Normaliza texto del Excel a una etiqueta canónica:
      HUMANO / IA / NO_SEGURO

    Regresa None si está vacío/NaN.
    """
    if resp is None:
        return None
    if isinstance(resp, float) and pd.isna(resp):
        return None

    text = str(resp).strip()
    if text == "":
        return None

    if text in RESPONSE_NORMALIZATION_MAP:
        return RESPONSE_NORMALIZATION_MAP[text]

    # Si hubiera variantes (ej. "Humano", "IA"), puedes extender aquí.
    # Por ahora, lo dejamos explícito para detectar datos inesperados.
    return f"UNKNOWN::{text}"


def load_authorship_long(
    excel_path: str,
    config: SurveyConfig = SurveyConfig(),
) -> pd.DataFrame:
    """
    Carga el Excel y devuelve un DataFrame long con respuestas de autoría.

    Columnas devueltas:
      participant_id, audio_index, response_raw, response_norm
    """
    df = pd.read_excel(excel_path)

    if config.id_col not in df.columns:
        # Si no existe ID, usa índice+1 como id.
        df = df.copy()
        df[config.id_col] = range(1, len(df) + 1)

    authorship_cols = _find_authorship_columns(list(df.columns), config.authorship_prefix)
    if not authorship_cols:
        raise ValueError(
            f"No se encontraron columnas de autoría con prefijo: {config.authorship_prefix}"
        )

    rows = []
    for _, row in df.iterrows():
        pid = row[config.id_col]
        for audio_idx, colname in authorship_cols:
            resp_raw = row[colname]
            resp_norm = _normalize_response(resp_raw)
            if resp_norm is None:
                continue
            rows.append(
                {
                    "participant_id": pid,
                    "audio_index": int(audio_idx),
                    "response_raw": resp_raw,
                    "response_norm": resp_norm,
                }
            )

    out = pd.DataFrame(rows)
    return out