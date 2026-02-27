"""
run_wilcoxon_utility.py
-----------------------
Lee Excel, localiza columna de "utilidad de IA" por needle y corre Wilcoxon vs 3.

Soporta respuestas:
- numéricas (1-5)
- textuales tipo Likert en español (Totalmente en desacuerdo ... Totalmente de acuerdo)
"""

from __future__ import annotations

import argparse
import pandas as pd

from analysis.categorical.io_categorical import _normalize_text
from analysis.ia_utility.wilcoxon_neutral import wilcoxon_vs_neutral


LIKERT_TEXT_TO_NUM = {
    # 1
    "totalmente en desacuerdo": 1,
    # 2
    "en desacuerdo": 2,
    # 3
    "ni de acuerdo ni en desacuerdo": 3,
    "neutral": 3,
    # 4
    "de acuerdo": 4,
    # 5
    "totalmente de acuerdo": 5,
}


def find_first_column_by_needle(columns, needle: str) -> str:
    needle_n = _normalize_text(needle)
    for col in columns:
        if needle_n in _normalize_text(str(col)):
            return col
    raise ValueError(f"No se encontró columna que contenga needle: {needle}")


def coerce_likert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convierte una serie de respuestas a valores 1-5.
    Acepta:
    - números (1..5)
    - strings Likert (mapeo arriba)
    """
    # Primero intenta numérico directo
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        return s_num

    # Si no hubo numéricos, intenta mapear texto
    def _map_one(x):
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        t = _normalize_text(str(x))
        if t == "":
            return None
        return LIKERT_TEXT_TO_NUM.get(t, None)

    mapped = series.apply(_map_one)
    return pd.to_numeric(mapped, errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wilcoxon vs neutral=3 (utilidad de IA)")
    parser.add_argument("--excel", required=True, help="Ruta al Excel")
    parser.add_argument("--needle", required=True, help="Subcadena del encabezado (p.ej. 'proceso creativo')")
    parser.add_argument("--neutral", type=float, default=3.0, help="Valor neutro (default 3)")
    parser.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    args = parser.parse_args()

    df = pd.read_excel(args.excel)
    col = find_first_column_by_needle(df.columns, args.needle)

    x = coerce_likert_to_numeric(df[col])

    # si sigue vacío, imprime valores únicos para debug
    if x.dropna().empty:
        uniques = df[col].dropna().astype(str).value_counts().head(30)
        raise ValueError(
            "No hay datos numéricos válidos tras mapear Likert.\n"
            f"col = {col}\n"
            "Ejemplos de valores en la columna:\n"
            f"{uniques.to_string()}"
        )

    res = wilcoxon_vs_neutral(x, neutral=args.neutral, alternative=args.alternative)

    print("\n=== Wilcoxon signed-rank vs neutral ===")
    print(f"col        = {col}")
    print(f"neutral    = {res.neutral}")
    print(f"statistic  = {res.statistic:.6f}")
    print(f"p_value    = {res.p_value:.12g}")
    print(f"n_total    = {res.n_total}")
    print(f"n_used     = {res.n_used} (excluye empates exactos con neutral)")


if __name__ == "__main__":
    main()