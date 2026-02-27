"""
build_dataset.py
----------------
Construye dataset para regresión logística global:

acierto ~ formación + horas_categoria
"""

import pandas as pd
import re
from analysis.turing.io_survey import load_authorship_long


AUDIO_TRUTH = {1: "IA", 2: "Humano", 3: "Humano", 4: "IA"}


import re
import pandas as pd

def parse_hours_to_number(x):
    if pd.isna(x):
        return None

    s = str(x).lower().strip()

    # 1) Formato tipo 1:30
    m = re.search(r"(\d+)\s*:\s*(\d+)", s)
    if m:
        h = float(m.group(1))
        mins = float(m.group(2))
        return h + mins / 60.0

    # 2) Si menciona "minuto(s)", convertir a horas
    if "minut" in s:
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            mins = float(m.group(1))
            return mins / 60.0

    # 3) Rangos tipo "7 a 9", "2-3", "3 o 4" -> promedio
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:a|-|o)\s*(\d+(?:\.\d+)?)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return (a + b) / 2.0

    # 4) Número simple
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))

    return None


def categorize_hours(x):
    if x is None:
        return "Missing"

    if x <= 2:
        return "0-2"
    elif x <= 5:
        return "3-5"
    else:
        return "6+"


def build_regression_df(excel_path):

    # Cargar autoría en formato long
    df_auth = load_authorship_long(excel_path)

    # Acierto
    df_auth["true_origin"] = df_auth["audio_index"].map(AUDIO_TRUTH)

    df_auth["acierto"] = (
        (df_auth["true_origin"] == "IA") & (df_auth["response_norm"] == "IA")
    ) | (
        (df_auth["true_origin"] == "Humano") & (df_auth["response_norm"] == "HUMANO")
    )

    df_auth["acierto"] = df_auth["acierto"].astype(int)

    # Cargar datos originales para formación y horas
    df_raw = pd.read_excel(excel_path)

    df_raw["participant_id"] = range(1, len(df_raw) + 1)

    df_raw["horas_num"] = df_raw["¿Cuántas horas al día escuchas música?"].apply(parse_hours_to_number)
    df_raw["horas_cat"] = df_raw["horas_num"].apply(categorize_hours)

    df_raw["formacion"] = df_raw["¿Tienes formación musical?"]

    # Merge
    df = df_auth.merge(
        df_raw[["participant_id", "horas_cat", "formacion"]],
        on="participant_id",
        how="left"
    )

    return df