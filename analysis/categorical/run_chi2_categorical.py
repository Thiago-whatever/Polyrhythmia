"""
run_chi2_categorical.py
-----------------------
Corre χ² de independencia para una variable categórica (por audio),
agrupando por origen real: IA (audios 1 y 4) vs Humano (audios 2 y 3).

Uso:
  python -m analysis.categorical.run_chi2_categorical --excel data/Results.xlsx --needle "<texto de la pregunta>"
"""

from __future__ import annotations

import argparse
from typing import Dict

import pandas as pd

from analysis.categorical.io_categorical import load_categorical_long
from analysis.turing.stats_chi2 import chi2_independence_test  # reutilizamos tu módulo


def default_audio_truth() -> Dict[int, str]:
    return {1: "IA", 2: "Humano", 3: "Humano", 4: "IA"}


def attach_truth(df_long: pd.DataFrame, truth: Dict[int, str]) -> pd.DataFrame:
    df = df_long.copy()
    df["true_origin"] = df["audio_index"].map(truth)
    if df["true_origin"].isna().any():
        missing = df[df["true_origin"].isna()]["audio_index"].unique().tolist()
        raise ValueError(f"Faltan audios en truth: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Chi² para variables categóricas (claridad/familiaridad/fusión)")
    parser.add_argument("--excel", required=True, help="Ruta al Excel")
    parser.add_argument("--needle", required=True, help="Subcadena para localizar la pregunta (columna)")
    args = parser.parse_args()

    df_long = load_categorical_long(args.excel, needle=args.needle)
    truth = default_audio_truth()
    df = attach_truth(df_long, truth)

    # Detecta valores UNKNOWN (si luego agregas value_map)
    unknown = df["value_norm"].astype(str).str.startswith("UNKNOWN::")
    if unknown.any():
        bad = df.loc[unknown, "value_raw"].value_counts()
        raise ValueError(
            "Se encontraron respuestas no reconocidas. "
            "Agrega un value_map para normalizar.\n"
            f"Valores:\n{bad.to_string()}"
        )

    # χ²: origen real (Humano/IA) vs categoría elegida
    # order_resp se deja dinámico: columnas según categorías presentes
    # (chi2_independence_test reindexa; aquí construimos directamente)
    res = chi2_independence_test(df, true_col="true_origin", resp_col="value_norm",
                                 order_true=["Humano", "IA"], order_resp=sorted(df["value_norm"].unique().tolist()))

    print("\n=== Tabla de contingencia (true_origin x value_norm) ===")
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


if __name__ == "__main__":
    main()