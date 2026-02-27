"""
run_logistic.py
---------------
Ejecuta regresión logística global.
"""

import argparse
import numpy as np

from analysis.regression.build_dataset import build_regression_df
from analysis.regression.logistic_model import fit_logistic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True)
    args = parser.parse_args()

    df = build_regression_df(args.excel)

    result = fit_logistic(df)

    print("\n=== Logistic Regression: acierto ~ formación + horas ===\n")
    print(result.summary())

    print("\n=== Odds Ratios ===")
    print(np.exp(result.params))


if __name__ == "__main__":
    main()