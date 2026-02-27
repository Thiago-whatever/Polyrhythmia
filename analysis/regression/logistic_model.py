"""
logistic_model.py
-----------------
Regresión logística global (statsmodels) con inputs 100% numéricos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_logistic(df: pd.DataFrame):
    df_model = df.copy()

    # Nos quedamos solo con lo necesario para el modelo
    keep = ["acierto", "formacion", "horas_cat"]
    missing = [c for c in keep if c not in df_model.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {list(df_model.columns)}")

    df_model = df_model[keep]

    # Limpieza básica: quitar NaNs en outcome o predictores
    df_model = df_model.dropna(subset=["acierto", "formacion", "horas_cat"]).copy()

    # Normaliza strings por si hay espacios raros
    df_model["formacion"] = df_model["formacion"].astype(str).str.strip()
    df_model["horas_cat"] = df_model["horas_cat"].astype(str).str.strip()

    # Dummies:
    # - formación: baseline = "Básica" (si existe)
    # - horas_cat: baseline = "0-2" (si existe)
    df_model = pd.get_dummies(df_model, columns=["formacion", "horas_cat"], drop_first=True)

    y = pd.to_numeric(df_model["acierto"], errors="raise").astype(float)

    X = df_model.drop(columns=["acierto"])

    # Asegurar que todo sea numérico
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop filas con NaN por coerción
    mask_ok = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask_ok].copy()
    y = y.loc[mask_ok].copy()

    X = sm.add_constant(X, has_constant="add").astype(float)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    return result