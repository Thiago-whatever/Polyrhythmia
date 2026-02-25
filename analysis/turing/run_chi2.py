"""
run_chi2.py
-----------
CLI: lee Excel -> construye dataframe long -> agrega ground truth -> corre χ².

Uso ejemplo:
  python analysis/turing/run_chi2.py --excel "data/tu_archivo.xlsx"
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import pandas as pd

from analysis.turing.io_survey import load_authorship_long
from analysis.turing.stats_chi2 import chi2_independence_test
from analysis.effect_sizes.cramers_v import cramers_v_from_table
from analysis.per_audio.binomial_audio import binomial_test_per_audio, results_to_dataframe


def default_audio_truth() -> Dict[int, str]:
    """
    Ground truth por tu diseño experimental:
      Audio 1 y 4 = IA
      Audio 2 y 3 = Humano
    """
    return {1: "IA", 2: "Humano", 3: "Humano", 4: "IA"}


def attach_truth(df_long: pd.DataFrame, truth: Dict[int, str]) -> pd.DataFrame:
    df = df_long.copy()
    df["true_origin"] = df["audio_index"].map(truth)
    missing = df["true_origin"].isna().sum()
    if missing > 0:
        raise ValueError(
            f"Hay {missing} filas sin ground truth. "
            f"Revisa que truth incluya todos los audio_index presentes."
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Chi-cuadrado de independencia (Turing autoria).")
    parser.add_argument("--excel", required=True, help="Ruta al archivo Excel con respuestas.")
    parser.add_argument(
        "--out_json",
        default=None,
        help="(Opcional) Ruta para guardar resultados en JSON.",
    )
    args = parser.parse_args()

    df_long = load_authorship_long(args.excel)

    # Detectar respuestas inesperadas
    unknown = df_long["response_norm"].astype(str).str.startswith("UNKNOWN::")
    if unknown.any():
        bad = df_long.loc[unknown, "response_raw"].value_counts()
        raise ValueError(
            "Se encontraron respuestas no reconocidas. "
            "Extiende RESPONSE_NORMALIZATION_MAP en io_survey.py.\n"
            f"Valores:\n{bad.to_string()}"
        )

    truth = default_audio_truth()
    df = attach_truth(df_long, truth)

    # χ² en tabla 2×3
    res = chi2_independence_test(df)

    print("\n=== Tabla de contingencia (true_origin x response_norm) ===")
    print(res.contingency_table)

    print("\n=== Chi-cuadrado de independencia ===")
    print(f"chi2 = {res.chi2:.6f}")
    print(f"dof  = {res.dof}")
    print(f"p    = {res.p_value:.12g}")

    print("\n=== Diagnóstico de supuestos (esperados) ===")
    print(f"min_expected      = {res.diagnostics.min_expected:.6f}")
    print(f"expected < 5      = {res.diagnostics.num_expected_lt_5}")
    print(f"expected < 1      = {res.diagnostics.num_expected_lt_1}")
    print(f"expected_shape    = {res.diagnostics.expected_shape}")

    # Tamaño de efecto: Cramér's V
    vres = cramers_v_from_table(res.contingency_table, res.chi2)
    print("\n=== Tamaño de efecto (Cramér's V) ===")
    print(f"V = {vres.v:.6f}   (n={vres.n}, shape={vres.table_shape})")

    # --- Binomial por audio ---
    print("\n=== Binomial por audio (3 opciones; p0=1/3; NO_SEGURO = incorrecto) ===")
    bin3 = binomial_test_per_audio(df_long, truth, mode="three_way")
    bin3_df = results_to_dataframe(bin3).sort_values("audio_index")
    print(bin3_df.to_string(index=False))

    print("\n=== Binomial por audio (2 opciones; excluye NO_SEGURO; p0=1/2) ===")
    bin2 = binomial_test_per_audio(df_long, truth, mode="two_way_exclude_no_seguro")
    bin2_df = results_to_dataframe(bin2).sort_values("audio_index")
    print(bin2_df.to_string(index=False))

    if args.out_json:
        payload = {
            "chi2": res.chi2,
            "dof": res.dof,
            "p_value": res.p_value,
            "contingency_table": res.contingency_table.to_dict(),
            "expected": res.expected.tolist(),
            "diagnostics": {
                "min_expected": res.diagnostics.min_expected,
                "num_expected_lt_5": res.diagnostics.num_expected_lt_5,
                "num_expected_lt_1": res.diagnostics.num_expected_lt_1,
                "expected_shape": list(res.diagnostics.expected_shape),
            },
            "cramers_v": {
                "V": vres.v,
                "n": vres.n,
                "k": vres.k,
                "table_shape": list(vres.table_shape),
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] JSON guardado en: {args.out_json}")


if __name__ == "__main__":
    main()