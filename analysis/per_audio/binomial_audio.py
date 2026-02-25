"""
binomial_audio.py
-----------------
Tests binomiales por audio para evaluar si la tasa de acierto es mayor
que el azar.

Para cada audio:
  - n = #respuestas válidas
  - k = #aciertos (respuesta coincide con el origen real)
  - H0: p = p0 (azar)
  - H1: p > p0 (one-sided; "mejor que azar")

Dos modos típicos:
  A) 3 opciones (Humano / IA / No seguro): p0 = 1/3.
     Aquí, "No seguro" cuenta como incorrecto (no es un acierto).
  B) 2 opciones (Humano / IA) colapsando/removiendo "No seguro":
     p0 = 1/2, y solo contamos respuestas HUMANO o IA.

Este archivo implementa ambos y te deja elegir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
from scipy.stats import binomtest


Mode = Literal["three_way", "two_way_exclude_no_seguro"]


@dataclass(frozen=True)
class BinomialAudioResult:
    audio_index: int
    true_origin: str
    mode: Mode
    n: int
    k_correct: int
    p0: float
    p_value_one_sided: float
    accuracy: float


def _is_correct(true_origin: str, response_norm: str) -> bool:
    """
    true_origin: "Humano" o "IA"
    response_norm: "HUMANO" / "IA" / "NO_SEGURO"
    """
    if true_origin == "Humano":
        return response_norm == "HUMANO"
    if true_origin == "IA":
        return response_norm == "IA"
    raise ValueError(f"true_origin inesperado: {true_origin}")


def binomial_test_per_audio(
    df_long: pd.DataFrame,
    truth_map: Dict[int, str],
    mode: Mode = "three_way",
) -> List[BinomialAudioResult]:
    """
    df_long debe tener:
      - audio_index (int)
      - response_norm ("HUMANO" / "IA" / "NO_SEGURO")

    truth_map: {audio_index: "Humano"|"IA"}

    mode:
      - "three_way": usa todas las respuestas; p0=1/3; NO_SEGURO cuenta incorrecto
      - "two_way_exclude_no_seguro": filtra NO_SEGURO; p0=1/2

    Retorna lista de resultados por audio (ordenada).
    """
    if mode == "three_way":
        p0 = 1.0 / 3.0
        df_use = df_long.copy()
    elif mode == "two_way_exclude_no_seguro":
        p0 = 0.5
        df_use = df_long[df_long["response_norm"].isin(["HUMANO", "IA"])].copy()
    else:
        raise ValueError(f"Modo no soportado: {mode}")

    results: List[BinomialAudioResult] = []

    for audio_idx in sorted(df_use["audio_index"].unique()):
        if int(audio_idx) not in truth_map:
            raise ValueError(f"audio_index {audio_idx} no tiene ground truth en truth_map")

        true_origin = truth_map[int(audio_idx)]

        sub = df_use[df_use["audio_index"] == audio_idx]
        n = int(len(sub))
        if n == 0:
            continue

        correct_mask = sub["response_norm"].apply(lambda r: _is_correct(true_origin, str(r)))
        k = int(correct_mask.sum())
        acc = k / n

        # Prueba binomial: H1 p > p0
        bt = binomtest(k, n, p=p0, alternative="greater")

        results.append(
            BinomialAudioResult(
                audio_index=int(audio_idx),
                true_origin=true_origin,
                mode=mode,
                n=n,
                k_correct=k,
                p0=p0,
                p_value_one_sided=float(bt.pvalue),
                accuracy=float(acc),
            )
        )

    return results


def results_to_dataframe(results: List[BinomialAudioResult]) -> pd.DataFrame:
    """
    Convierte lista de resultados a DataFrame.
    """
    return pd.DataFrame(
        [
            {
                "audio_index": r.audio_index,
                "true_origin": r.true_origin,
                "mode": r.mode,
                "n": r.n,
                "k_correct": r.k_correct,
                "accuracy": r.accuracy,
                "p0": r.p0,
                "p_value_one_sided": r.p_value_one_sided,
            }
            for r in results
        ]
    )