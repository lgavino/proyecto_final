from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "data.csv"

NUMERIC_COLS = ["Age", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a número las columnas numéricas; lo no convertible queda como NaN."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Sin dolor (Pain = 0) significa NRS_pain = 0, no que falte el dato
    if "Pain" in df.columns and "NRS_pain" in df.columns:
        df.loc[df["Pain"] == 0, "NRS_pain"] = 0
    return df


def load_default_data() -> pd.DataFrame:
    """Carga el CSV incluido en el proyecto."""
    df = pd.read_csv(DEFAULT_CSV, sep=";", encoding="latin-1")
    return _clean(df)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Carga un CSV enviado desde Streamlit."""
    df = pd.read_csv(uploaded_file, sep=";", encoding="latin-1")
    return _clean(df)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Resume dimensiones, duplicados, nulos y variables numéricas."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
    }