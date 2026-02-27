"""
run_likert.py
-------------
CLI para:
- Cargar Likert por audio
- Adjuntar truth (1 y 4 IA; 2 y 3 Humano)
- Reportar mediana + IQR
- Mann–Whitney U (IA vs Humano)
- (Opcional) Kruskal–Wallis por audio

Uso:
  python -m analysis.likert.run_likert --excel data/Results.xlsx
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

from analysis.likert.io_likert import load_likert_long
from analysis.likert.stats_likert import (
    attach_truth_origin,
    summary_by_origin,
    summary_by_audio,
    mann_whitney_ia_vs_humano,
    kruskal_by_audio,
)


def default_audio_truth() -> Dict[int, str]:
    return {1: "IA", 2: "Humano", 3: "Humano", 4: "IA"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Likert analysis: median/IQR + MWU + optional Kruskal")
    parser.add_argument("--excel", required=True, help="Ruta al Excel con respuestas")
    parser.add_argument("--out_json", default=None, help="(Opcional) Guardar resultados en JSON")
    parser.add_argument(
        "--no_kruskal",
        action="store_true",
        help="Si se incluye, NO corre Kruskal–Wallis por audio",
    )
    args = parser.parse_args()

    df_long = load_likert_long(args.excel)
    truth = default_audio_truth()
    df = attach_truth_origin(df_long, truth)

    print("\n=== Likert: resumen por origen (mediana + IQR) ===")
    s_origin = summary_by_origin(df)
    print(s_origin.to_string(index=False))

    print("\n=== Likert: resumen por audio (mediana + IQR) ===")
    s_audio = summary_by_audio(df_long)
    print(s_audio.to_string(index=False))

    print("\n=== Mann–Whitney U: IA vs Humano (two-sided) ===")
    mwu = mann_whitney_ia_vs_humano(df)
    print(f"U = {mwu.u:.6f}")
    print(f"p = {mwu.p_value_two_sided:.12g}")
    print(f"n_ia = {mwu.n_ia}, n_humano = {mwu.n_humano}")

    kr = None
    if not args.no_kruskal:
        print("\n=== Kruskal–Wallis: comparación por audio (1..4) ===")
        kr = kruskal_by_audio(df_long)
        print(f"H = {kr.h:.6f}")
        print(f"p = {kr.p_value:.12g}")
        print(f"groups = {kr.groups}, n_total = {kr.n_total}")

    if args.out_json:
        payload = {
            "summary_by_origin": s_origin.to_dict(orient="records"),
            "summary_by_audio": s_audio.to_dict(orient="records"),
            "mann_whitney": {
                "U": mwu.u,
                "p_value_two_sided": mwu.p_value_two_sided,
                "n_ia": mwu.n_ia,
                "n_humano": mwu.n_humano,
            },
        }
        if kr is not None:
            payload["kruskal_wallis"] = {
                "H": kr.h,
                "p_value": kr.p_value,
                "groups": kr.groups,
                "n_total": kr.n_total,
            }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] JSON guardado en: {args.out_json}")


if __name__ == "__main__":
    main()